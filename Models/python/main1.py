from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import logging
import argparse
import signal
import sys
from typing import List

from communication import UnityCommunicationServer
from monitor import RewardMonitor
from logging.handlers import RotatingFileHandler
from visualizer import plot_network
from mlp_model import build_mlp, train_mlp
from cnn_model import build_cnn, train_cnn
from data_processing import load_for_mlp, load_for_cnn, Preproc, Optional
from rl_agent import RLAgent
import numpy as np
import torch

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the locomotion server with offline and on‑line RL components")
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'],
                        help="The base model to use for offline training")
    parser.add_argument('--host', type=str, default='127.0.0.1', help='IP address to bind the server')
    parser.add_argument('--port', type=int, default=6900, help='TCP port to listen on')
    # Hidden layers can be specified as a comma‑separated list, e.g. "128,64"
    parser.add_argument('--hidden_layers', type=str, default='128,64',
                        help='Comma‑separated sizes of hidden layers for the MLP and RL networks')
    parser.add_argument('--input_size', type=int, default=10,
                        help="Number of observation features (raw angles) per time step")
    parser.add_argument('--output_size', type=int, default=None,
                        help="Number of action outputs (degrees of freedom). Defaults to input_size if not set.")
    parser.add_argument('--preproc_method', type=str, default='rad', choices=['rad', 'sin_cos'],
                        help="Preprocessing method for input angles: 'rad' (radians) or 'sin_cos' to double the dimensionality")
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs for offline supervised training')
    parser.add_argument('--rl_lr', type=float, default=1e-4,
                        help='Learning rate for the on‑line RL agent')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for the RL agent')
    parser.add_argument('--use_rl', action='store_true', default=True,
                        help='Enable on‑line reinforcement learning using the RLAgent after offline training')
    parser.add_argument('--dt', type=int, default=20, help='Simulated time per tick (milliseconds)')
    parser.add_argument('--rewards_size', type=int, default=2,
                        help='Number of previous rewards kept for plateu detection (passed to Unity)')
    parser.add_argument('--patience_max', type=int, default=30,
                        help='Number of failed checks before adding random rewards to escape plateus (passed to Unity)')
    return parser.parse_args()

_step = 0

def main() -> None:
    args = parse_args()
    # Parse hidden layer configuration
    hidden_layers = [int(h) for h in args.hidden_layers.split(',')] if args.hidden_layers else [128, 64]
    # Default output_size to input_size if not specified
    if args.output_size is None:
        args.output_size = args.input_size
    # Configure preprocessing of angles
    cfg = Preproc(rep=args.preproc_method, unwrap=True, seq_len=8)
    # Load offline training data; here we use MLP loader for simplicity
    # Modify the filename as needed to point to your captured animation data
    train_dl, val_dl = load_for_mlp("animation_data_20251006_142301.json", cfg)
    # Build and train the offline model (supervised)
    if args.model.lower() == 'mlp':
        model = build_mlp(args.input_size, args.output_size, hidden_layers)
        train_mlp(model, train_dl, val_dl, epochs=args.epochs)
    elif args.model.lower() == 'cnn':
        model = build_cnn(args.input_size, args.output_size, feature_len_hint=cfg.seq_len)
        train_cnn(model, train_dl, val_dl, epochs=args.epochs)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Set up reward monitoring (log moving averages)
    reward_monitor = RewardMonitor(window_size=100)

    # Place‑holder for last observation across calls; used by RLAgent to compute TD updates.
    last_obs: Optional[np.ndarray] = None
    # Optionally wrap the offline model with an RL agent for on‑line adaptation
    if args.use_rl:
        # RL agent uses the same hidden architecture as the offline model
        rl_agent = RLAgent(
            obs_dim=args.input_size,
            action_dim=args.output_size,
            hidden_sizes=hidden_layers,
            lr=args.rl_lr,
            gamma=args.gamma,
        )
        # Initialise actor weights from the offline model
        rl_agent.load_from_model(model)

    def handle_message(values: List[float]) -> List[float]:
        """
        Process a message from Unity and return action commands.

        Parameters
        ----------
        values : list[float]
            The incoming data from Unity.  It should contain a number
            of floats corresponding to raw joint observations,
            followed by optional foot contact flags and reward/error
            terms.  The exact layout depends on how you pack your
            features in Unity.  Here we assume that ``args.input_size``
            raw observation features are sent first, and any
            additional floats are treated as reward and error signals.

        Returns
        -------
        list[float]
            The action vector to send back to Unity.  Its length
            equals ``args.output_size``.
        """
        nonlocal last_obs  # used only if args.use_rl is true
        # Extract observation and reward/error.  We assume the first
        # ``args.input_size`` entries are observation features.  Any
        # additional entries are interpreted as [reward, error, …].  If
        # less than input_size values are provided, pad with zeros.
        obs_raw = values[: args.input_size]
        # Pad with zeros if fewer than expected
        if len(obs_raw) < args.input_size:
            obs_raw = obs_raw + [0.0] * (args.input_size - len(obs_raw))
        # Remaining values after obs are reward and error signals
        reward = 0.0
        err = 0.0
        if len(values) > args.input_size:
            reward = float(values[-2])
        if len(values) > args.input_size + 1:
            err = float(values[-1])
        # Preprocess observations according to the selected method
        # convert_obs is handled inside load_for_mlp when offline; for on-line we
        # apply the same transformation here
        # For 'rad', raw observations are assumed to already be in radians.
        # For 'sin_cos', we convert each angle to sin and cos representation.
        if args.preproc_method == 'sin_cos':
            # Each raw angle expands to two features (sin and cos)
            # If obs_raw length is N, obs becomes length 2*N
            angles = np.array(obs_raw, dtype=np.float32)
            sin_comp = np.sin(angles)
            cos_comp = np.cos(angles)
            obs = np.concatenate([sin_comp, cos_comp], axis=0).astype(np.float32)
        else:
            # 'rad' mode: use raw angles directly as features
            obs = np.array(obs_raw, dtype=np.float32)
        # Offline MLP produces deterministic actions; RL agent adds on‑line adaptation
        if args.use_rl:
            # Update the agent using the reward/error for the previous step
            # Combine reward and error into a single scalar reward (penalise error)
            reward_total = reward # - err # The error is oversaturating the reward.
            # If last_obs exists, perform an update with the new observation
            if last_obs is not None:
                rl_agent.update(reward_total, obs, done=False)
            # Select a new action
            action, logp, val = rl_agent.select_action(obs)
            # Store current obs for next update
            last_obs = obs
            # Monitor rewards (for logging only, not used by RL agent)
            reward_monitor.record(reward_total)
            # Clamp actions to [-1, 1] to avoid extreme values
            actions = np.clip(action, -1.0, 1.0).tolist()
        else:
            # Pure offline inference: feed obs through the MLP/CNN to get actions
            import torch
            model.eval()
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                preds = model(x).squeeze(0)
                actions = preds.cpu().numpy().tolist()
        return actions

    # Create and start the communication server
    server = UnityCommunicationServer(
        host=args.host,
        port=args.port,
        dt=args.dt,
        rewards_size=args.rewards_size,
        patience_max=args.patience_max,
    )

    def handle_sigint(signum, frame) -> None:
        print("\n[Main] Received signal to stop, shutting down…")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    try:
        server.start(handle_message)
    except KeyboardInterrupt:
        handle_sigint(None, None)