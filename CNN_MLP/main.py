from __future__ import annotations

"""Entry point for the locomotion server.

This script can train an MLP or CNN on recorded animation data and then
start a simple TCP server that receives observation vectors from a
Unity simulation and returns predicted action vectors.  Optionally it
records reward and error signals and logs moving averages over a
sliding window.
"""

import argparse
import signal
import sys
import torch
import numpy as np
from typing import List, Deque
from collections import deque

from communication import UnityCommunicationServer
from monitor import RewardMonitor, ErrorMonitor
from mlp_model import build_mlp, train_mlp
from cnn_model import build_cnn, train_cnn
from data_processing import load_for_mlp, load_for_cnn, Preproc

def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    The default values are chosen to work with the provided animation
    data set.  You can override them to experiment with different
    models, sequence lengths or learning rates.
    """
    parser = argparse.ArgumentParser(description="Run the locomotion server (MLP or CNN)")
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'],
                        help="The model architecture to use: 'mlp' or 'cnn'")
    parser.add_argument('--data', type=str, default='animation_data_20251006_154939.json',
                        help="Path to the JSON animation data file")
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='IP address to bind the server (Unity should connect here)')
    parser.add_argument('--port', type=int, default=6900,
                        help='TCP port to listen on')
    parser.add_argument('--input_size', type=int, default=12,
                        help="Number of features per observation (should match data)")
    parser.add_argument('--output_size', type=int, default=12,
                        help="Number of actions to output (defaults to same as input)")
    parser.add_argument('--hidden_layers', type=str, default='128,64',
                        help="Comma‑separated sizes of hidden layers for the MLP")
    parser.add_argument('--seq_len', type=int, default=8,
                        help="Sequence length for the CNN (number of past frames)")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of epochs for offline training")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate for offline training")
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Batch size for offline training")
    parser.add_argument('--dt', type=int, default=20,
                        help='Simulated time per tick (sent to Unity)')
    parser.add_argument('--rewards_size', type=int, default=1,
                        help='Number of previous rewards sent in the handshake')
    parser.add_argument('--patience_max', type=int, default=30,
                        help='Patience parameter sent in the handshake')
    return parser.parse_args()

_step = 0

def main() -> None:
    """Main entry point.

    Performs optional offline training on the provided dataset and
    launches a communication server that wraps the trained model.  The
    server reads observation vectors from Unity, predicts actions and
    returns them.  When reward and error signals are provided, they
    are recorded via monitors for later analysis.
    """
    args = parse_args()

    # Configure preprocessing for the data loaders
    cfg = Preproc(rep="rad", unwrap=True, seq_len=args.seq_len)

    # Offline training
    model = None
    if args.model == 'mlp':
        # Load data for MLP (single frame -> next frame)
        train_dl, val_dl = load_for_mlp(args.data, cfg)
        hidden_sizes = [int(s) for s in args.hidden_layers.split(',') if s]
        model = build_mlp(args.input_size, args.output_size, hidden_layers=hidden_sizes)
        print(f"[Main] Training MLP with {hidden_sizes} hidden units…")
        train_mlp(model, train_dl, val_dl, epochs=args.epochs, lr=args.lr)
    elif args.model == 'cnn':
        # Load data for CNN (sequence -> next frame)
        train_dl, val_dl = load_for_cnn(args.data, cfg)
        model = build_cnn(args.input_size, args.output_size)
        print(f"[Main] Training CNN with seq_len={args.seq_len}…")
        train_cnn(model, train_dl, val_dl, epochs=args.epochs, lr=args.lr)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Monitors for rewards and errors
    reward_monitor = RewardMonitor(window_size=100)
    error_monitor = ErrorMonitor(window_size=100)

    # Buffer for CNN: maintain a sliding window of past observations
    history: Deque[List[float]] = deque(maxlen=args.seq_len)

    def handle_message(values: List[float]) -> List[float]:
        """Process a list of floats received from Unity.

        The expected message format is either:
        - ``obs`` of length ``input_size``
        - ``obs`` + reward + error (length ``input_size + 2``)

        The function returns a list of floats of length ``output_size``.
        """
        # Extract observation, reward and error from the message
        if len(values) == args.input_size + 2:
            obs = values[:-2]
            reward = values[-2]
            err = values[-1]
        elif len(values) == args.input_size:
            obs = values
            reward = 0.0
            err = 0.0
        else:
            # Pad or truncate to expected input size
            obs = values[:args.input_size]
            reward = 0.0
            err = 0.0

        # Record reward and error signals
        if reward != 0.0:
            reward_monitor.record(reward)
            if len(reward_monitor.all_rewards) % 100 == 0:
                print(f"[Monitor] Reward moving average ({reward_monitor.window_size}) = {reward_monitor.moving_average():.4f}")
        if err != 0.0:
            error_monitor.record(err)
            if len(error_monitor.all_errors) % 100 == 0:
                print(f"[Monitor] Error moving average ({error_monitor.window_size}) = {error_monitor.moving_average():.4f}")

        # Compute the action vector
        if args.model == 'mlp':
            # For MLP, single frame input
            tensor_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            tensor_obs = tensor_obs.to(next(model.parameters()).device)
            with torch.no_grad():
                actions = model(tensor_obs).cpu().squeeze(0).tolist()
        else:
            # For CNN, accumulate sequence into history buffer
            history.append(obs)
            if len(history) < args.seq_len:
                # Not enough frames yet; return zeros
                actions = [0.0] * args.output_size
            else:
                # Build tensor shape (1, C, L)
                arr = np.array(history, dtype=np.float32).T  # (C,L)
                tensor_seq = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
                tensor_seq = tensor_seq.to(next(model.parameters()).device)
                with torch.no_grad():
                    actions = model(tensor_seq).cpu().squeeze(0).tolist()

        return actions

    # Create and start the communication server
    server = UnityCommunicationServer(host=args.host, port=args.port,
                                     dt=args.dt, rewards_size=args.rewards_size,
                                     patience_max=args.patience_max)

    def handle_sigint(signum, frame) -> None:
        """Handle SIGINT to stop server gracefully."""
        print("\n[Main] Received signal to stop, shutting down…")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    try:
        server.start(handle_message)
    except KeyboardInterrupt:
        handle_sigint(None, None)

main()