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

import argparse
import signal
import sys
from typing import List

from snn_model import SpikingNetwork
from communication import UnityCommunicationServer
from monitor import RewardMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SNN locomotion server")
    parser.add_argument('--host', type=str, default='127.0.0.1', help='IP address to bind the server')
    parser.add_argument('--port', type=int, default=6900, help='TCP port to listen on')
    parser.add_argument('--hidden-size', type=int, default=300, help='Number of hidden neurons')
    parser.add_argument('--cerebellum-size', type=int, default=100, help='Number of cerebellum neurons')
    parser.add_argument('--dt', type=int, default=5, help='Simulated time per tick')
    parser.add_argument('--rewards_size', type=int, default=2, help='Number of previous rewards kept for linear fitting to recognising learning plateus')
    parser.add_argument('--patience_max', type=int, default=30, help='Number of failed checks before adding random rewards to escape plateus')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Initialise the spiking network.  Hidden size can be tuned via CLI.
    network = SpikingNetwork(
        host=args.host,
        port=args.port,
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
        # Determine whether a reward is present
        if len(values) == args.input_size + 1:
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
