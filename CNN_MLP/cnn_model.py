"""
cnn_model.py
------------
Simple 1D-CNN for time-series joint data + training loop.

This file provides:
- SimpleCNN (nn.Module)  -> 2 conv blocks + global flatten + Linear
- build_cnn(...)         -> convenience constructor
- train_cnn(...)         -> training loop identical in spirit to train_mlp
- _evaluate(...)         -> shared evaluation helper

Expected input
--------------
- Tensors with shape (B, C, T):
  B = batch size, C = channels (e.g., joints * per-joint features), T = time steps.
"""

from __future__ import annotations

from typing import Optional, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class SimpleCNN(nn.Module):
    """
    Tiny 1D-CNN over temporal axis.

    Architecture
    ------------
    Input: (B, C, T)
      Conv1d(C -> 64, k=3, pad=1) -> ReLU -> MaxPool1d(2)
      Conv1d(64 -> 128, k=3, pad=1) -> ReLU -> MaxPool1d(2)
      Flatten -> Linear(128 * T' -> out_dim)
    where T' is the time length after two poolings: roughly T/4 (integer floor).

    Notes
    -----
    - We create the final Linear lazily because T can vary between datasets.
    - If you know T' ahead of time, you can precompute the `in_features` and
      replace the lazy creation with a fixed Linear.
    """

    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        # First temporal conv: keeps length with padding=1 (k=3)
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        # Second temporal conv
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        # Shared max pool (downsample by 2 along T)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        # We'll initialize the final classifier/regressor on the first forward pass,
        # when we know the flattened feature dimension (128 * T').
        self._fc: Optional[nn.Linear] = None
        self._out_dim = out_dim

    def _ensure_fc(self, x: torch.Tensor) -> None:
        """
        Create the final Linear layer lazily the first time we see an input.
        """
        if self._fc is None:
            # x shape after conv/pool stack: (B, 128, T')
            flat = x.shape[1] * x.shape[2]  # 128 * T'
            self._fc = nn.Linear(flat, self._out_dim).to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        - Conv -> ReLU -> Pool
        - Conv -> ReLU -> Pool
        - Flatten -> Linear
        """
        # Ensure input is (B, C, T)
        # If your data is (B, T, C), swap dims before calling: x = x.transpose(1, 2)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        self._ensure_fc(x)
        x = x.flatten(1)  # flatten all but batch dim
        return self._fc(x)


def build_cnn(input_channels: int, output_dim: int) -> nn.Module:
    """
    Convenience wrapper to create SimpleCNN.

    Parameters
    ----------
    input_channels : int
        Number of channels in input (e.g., joints * features_per_joint).
    output_dim : int
        Size of output vector (regression dim or #classes).
    """
    return SimpleCNN(input_channels, output_dim)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Mean loss over a DataLoader without gradient updates.
    """
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        loss = criterion(preds, y)
        bs = x.size(0)
        total += loss.item() * bs
        count += bs
    return total / max(count, 1)


def train_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    criterion: Optional[nn.Module] = None,
    optimizer_cls = torch.optim.Adam,
    device: Optional[torch.device] = None,
    grad_clip: Optional[float] = None,
    log_interval: int = 50,
) -> Dict[str, float]:
    """
    Minimal training loop (mirrors train_mlp) with clear, safe defaults.

    For classification:
      - set `criterion=nn.CrossEntropyLoss()`
      - ensure targets are dtype torch.long of shape (B,)
      - set model final layer `out_dim` to #classes
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = criterion or nn.MSELoss()
    opt = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None
    step = 0

    for ep in range(1, epochs + 1):
        model.train()
        running, count = 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # --- standard training step ---
            opt.zero_grad(set_to_none=True)    # clear old grads
            preds = model(xb)                   # forward
            loss = criterion(preds, yb)         # compute loss
            loss.backward()                     # backprop
            if grad_clip is not None:           # optional stabilization
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()                          # update weights

            # --- bookkeeping/logging ---
            bs = xb.size(0)
            running += loss.item() * bs
            count += bs
            step += 1
            if log_interval and step % log_interval == 0:
                print(f"[ep {ep:03d} step {step:06d}] "
                      f"train_loss={running/max(count,1):.4f}")

        train_loss = running / max(count, 1)

        if val_loader is not None:
            val_loss = _evaluate(model, val_loader, criterion, device)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_d_
