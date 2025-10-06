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
from data_processing import load_for_mlp, load_for_cnn, Preproc

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SNN locomotion server")
    parser.add_argument('--model', type=str, default='mlp', help="The model to use. Can be MLP or CNN")
    parser.add_argument('--host', type=str, default='127.0.0.1', help='IP address to bind the server')
    parser.add_argument('--port', type=int, default=6900, help='TCP port to listen on')
    parser.add_argument('--hidden_size', type=int, default=1500, help='Number of hidden neurons')
    parser.add_argument('--cerebellum_size', type=int, default=850, help='Number of cerebellum neurons')
    parser.add_argument('--input_size', type=int, default=10, help="The amount of features, excluding rewards")
    parser.add_argument('--dt', type=int, default=20, help='Simulated time per tick')
    parser.add_argument('--rewards_size', type=int, default=2, help='Number of previous rewards kept for linear fitting to recognising learning plateus')
    parser.add_argument('--patience_max', type=int, default=30, help='Number of failed checks before adding random rewards to escape plateus')
    return parser.parse_args()

_step = 0

def main() -> None:
    args = parse_args()

    cfg = Preproc(rep="rad", unwrap=True, seq_len=8)

    train_dl, val_dl = load_for_mlp("animation_data_20251006_142301.json", cfg)

    model = None
    if (args.model == 'MLP'):
        model = build_mlp(args.input_size, args.output_size, args.hidden_size)
        train_mlp(model, train_dl, val_dl, epochs=50)
    elif (args.model == 'CNN'):
        model = build_cnn(args.input_size, args.output_size)
        train_cnn()



    reward_monitor = RewardMonitor(window_size=100)

    def handle_message(values: List[float]) -> List[float]:
        """
        Process a list of floats received from Unity.

        If the length of ``values`` equals ``input_size + 1`` we
        interpret the last element as a reward signal.  Otherwise
        reward defaults to zero.  The remainder of the list is
        treated as observation features and passed to the model.

        The returned list of floats has length ``output_size``.
        """
        global _step
        _step += 1

        # Determine whether a reward is present
        if len(values) == args.input_size + 2:
            obs = values[:-2]
            reward = values[-2]
            error = values[-1]
        elif len(values) == args.input_size:
            obs = values
            reward = 0.0
            error = 0.0
        else:
            # Unexpected length; pad or truncate as needed
            obs = values[: args.input_size]
            reward = 0.0
            error = 0.0

        # Apply reward to update weights of the hidden layer
        if reward != 0.0:
            model.set_reward(reward)
            reward_monitor.record(reward)
            if len(reward_monitor.all_rewards) % 100 == 0:
                avg = reward_monitor.moving_average()
                print(f"[Monitor] Reward moving average over {reward_monitor.window_size}: {avg:.3f}")
        # Apply error to update weights of cerebellum
        if error != 0.0:
            model.set_error(error)
            reward_monitor.record(error)
            if len(reward_monitor.all_rewards) % 100 == 0:
                avg = reward_monitor.moving_average()
                print(f"[Monitor] Error moving average over {reward_monitor.window_size}: {avg:.3f}")

        actions = model.step(obs)

        return actions

    # Create and start the communication server
    server = UnityCommunicationServer(host=args.host, port=args.port, dt=args.dt, rewards_size=args.rewards_size, patience_max=args.patience_max)

    def handle_sigint(signum, frame) -> None:
        print("\n[Main] Received signal to stop, shutting down…")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    try:
        server.start(handle_message) # Starts the model
    except KeyboardInterrupt:
        handle_sigint(None, None)