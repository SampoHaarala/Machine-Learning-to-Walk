"""Monitoring utilities for reward and error tracking.

This module defines simple classes to record and report the progress of
learning.  Use these monitors in training loops to log
rewards, prediction errors, or other metrics.  They support
computing moving averages and saving data to CSV files for later
analysis.
"""

from __future__ import annotations

import csv
import time
from collections import deque
from typing import Deque, Iterable, List, Optional, Tuple


class RewardMonitor:
    """Record and summarise reward values over time."""

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.rewards: Deque[float] = deque(maxlen=window_size)
        self.timestamps: Deque[float] = deque(maxlen=window_size)
        self.all_rewards: List[Tuple[float, float]] = []  # (timestamp, reward)

    def record(self, reward: float) -> None:
        """Record a reward value with the current timestamp."""
        now = time.time()
        self.rewards.append(reward)
        self.timestamps.append(now)
        self.all_rewards.append((now, reward))

    def moving_average(self) -> float:
        """Return the moving average over the last ``window_size`` rewards."""
        if not self.rewards:
            return 0.0
        return sum(self.rewards) / len(self.rewards)

    def save_csv(self, filename: str) -> None:
        """Save the recorded rewards to a CSV file.
        Each row contains a timestamp and the corresponding reward value.
        """
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'reward'])
            for ts, r in self.all_rewards:
                writer.writerow([ts, r])


class ErrorMonitor:
    """Track prediction errors (e.g. reward prediction error)."""

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.errors: Deque[float] = deque(maxlen=window_size)
        self.all_errors: List[Tuple[float, float]] = []  # (timestamp, error)

    def record(self, error: float) -> None:
        now = time.time()
        self.errors.append(error)
        self.all_errors.append((now, error))

    def moving_average(self) -> float:
        if not self.errors:
            return 0.0
        return sum(self.errors) / len(self.errors)

    def save_csv(self, filename: str) -> None:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'error'])
            for ts, e in self.all_errors:
                writer.writerow([ts, e])
