"""
Entry point for the SNN server.

This script instantiates a spiking neural network, sets up a TCP
server for communication with Unity and processes incoming
observations to produce control outputs.  It also logs reward
signals to monitor learning progress.

Run this script with default settings using

    python3 main.py

or specify a custom host/port:

    python3 main.py --host 0.0.0.0 --port 9001
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import logging
import argparse
import signal
import sys
from typing import List

from snn_model import SpikingNetwork
from communication import UnityCommunicationServer
from monitor import RewardMonitor
from logging.handlers import RotatingFileHandler
from visualizer import plot_network

logger = logging.getLogger("snn")
logger.setLevel(logging.INFO)

# Console
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

# Rotating file (optional)
fh = RotatingFileHandler("snn.log", maxBytes=2_000_000, backupCount=3)
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

logger.handlers.clear()
logger.addHandler(ch)
logger.addHandler(fh)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SNN locomotion server")
    parser.add_argument('--host', type=str, default='127.0.0.1', help='IP address to bind the server')
    parser.add_argument('--port', type=int, default=6900, help='TCP port to listen on')
    parser.add_argument('--hidden-size', type=int, default=1500, help='Number of hidden neurons')
    parser.add_argument('--cerebellum-size', type=int, default=850, help='Number of cerebellum neurons')
    parser.add_argument('--input_size', type=int, default=16, help="The amount of features, excluding rewards")
    parser.add_argument('--dt', type=int, default=20, help='Simulated time per tick')
    parser.add_argument('--rewards_size', type=int, default=2, help='Number of previous rewards kept for linear fitting to recognising learning plateus')
    parser.add_argument('--patience_max', type=int, default=30, help='Number of failed checks before adding random rewards to escape plateus')
    return parser.parse_args()

_step = 0

def main() -> None:
    args = parse_args()

    # Initialise the spiking network.  Hidden size can be tuned via CLI.
    network = SpikingNetwork(
        host=args.host,
        port=args.port,
        input_size=args.input_size + 2,
        N_h=args.hidden_size,
        N_c=args.cerebellum_size
    )

    reward_monitor = RewardMonitor(window_size=100)

    def handle_message(values: List[float]) -> List[float]:
        """
        Process a list of floats received from Unity.

        If the length of ``values`` equals ``input_size + 1`` we
        interpret the last element as a reward signal.  Otherwise
        reward defaults to zero.  The remainder of the list is
        treated as observation features and passed to the network.

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
            network.set_reward(reward)
            reward_monitor.record(reward)
            if len(reward_monitor.all_rewards) % 100 == 0:
                avg = reward_monitor.moving_average()
                print(f"[Monitor] Reward moving average over {reward_monitor.window_size}: {avg:.3f}")
        # Apply error to update weights of cerebellum
        if error != 0.0:
            network.set_error(error)
            reward_monitor.record(error)
            if len(reward_monitor.all_rewards) % 100 == 0:
                avg = reward_monitor.moving_average()
                print(f"[Monitor] Error moving average over {reward_monitor.window_size}: {avg:.3f}")

        # Compute actions via one simulation step
        actions = network.step(obs)

        if _step % 1000 == 0:
            fig = plot_network(network)
            fig.savefig(f"checkpoints/network_{_step:06d}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
        
        if _step % 1000 == 0 and reward_monitor.all_rewards:
            ts, rs = zip(*reward_monitor.all_rewards[-100:])  # last 5k points
            fig, ax = plt.subplots(figsize=(6,3))
            ax.plot(rs, linewidth=1)
            ax.set_title("Reward"); ax.set_xlabel("step"); ax.set_ylabel("r")
            fig.savefig(f"checkpoints/reward_{_step:06d}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
        
        if _step % 20 == 0:
            logger.info("step=%d rx_n=%d tx_n=%d rx_head=%s tx_head=%s reward=%.3f error=%.3f",_step, len(values), len(actions), values[:4], actions[:4], reward, error)

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


if __name__ == '__main__':
    main()
