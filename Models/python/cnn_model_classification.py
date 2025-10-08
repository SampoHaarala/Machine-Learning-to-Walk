"""
Modified 1D‑CNN model and training loop for both regression and classification.

This module provides a simple convolutional network ``SimpleCNN`` for
time‑series joint data along with helper functions ``build_cnn`` and
``train_cnn``. The network architecture is unchanged from the original
implementation: two 1D convolutions along the temporal axis followed by
max pooling and a fully connected layer. The training loop has been
extended to support classification by specifying ``classification=True``
when invoking ``train_cnn``. An optional ``binary`` flag allows
distinguishing between binary (two‑class) and multi‑class problems.

When performing classification:

* The loss defaults to ``BCEWithLogitsLoss`` for binary tasks and
  ``CrossEntropyLoss`` for multi‑class tasks.
* Targets are cast to the appropriate dtype: ``float`` for binary
  classification (with values 0/1) and ``long`` for multi‑class.
* Training and validation accuracies are computed and returned.

For regression the behaviour remains the same as the original: mean
squared error (MSE) is used by default and the output dimension
corresponds to the regression dimensionality.
"""

from __future__ import annotations

from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class SimpleCNN(nn.Module):
    """
    Tiny 1D‑CNN over a temporal axis.

    Architecture
    ----------
    Input: (B, C, T)
    Conv1d(C → 64, k=3, pad=1) → ReLU → MaxPool1d(2)
    Conv1d(64 → 128, k=3, pad=1) → ReLU → MaxPool1d(2)
    Flatten → Linear(128 * T' → out_dim) where T' ≈ T/4 after pooling.

    The final fully connected layer is created lazily on the first
    forward pass so that it adapts to the temporal dimension of the
    data. ``out_dim`` should be set equal to the number of regression
    outputs or classes depending on the task. For binary
    classification ``out_dim`` should be 1.
    """

    def __init__(self, in_channels: int, out_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self._fc: Optional[nn.Linear] = None
        self._out_dim = out_dim

    def _ensure_fc(self, x: torch.Tensor) -> None:
        if self._fc is None:
            flat = x.shape[1] * x.shape[2]
            self._fc = nn.Linear(flat, self._out_dim).to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input shape (B, C, T). If (B, T, C) is provided the user
        # should transpose prior to calling this method.
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        self._ensure_fc(x)
        x = x.flatten(1)
        return self._fc(x)


def build_cnn(input_channels: int, output_dim: int) -> nn.Module:
    """Convenience wrapper to create ``SimpleCNN``.

    Parameters
    ----------
    input_channels : int
        Number of input channels (e.g., joints * features_per_joint).
    output_dim : int
        Size of the network output. For classification this equals the
        number of classes (use 1 for binary classification).

    Returns
    -------
    nn.Module
        The ``SimpleCNN`` instance.
    """
    return SimpleCNN(input_channels, output_dim)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    classification: bool = False,
    binary: bool = False,
) -> Dict[str, float]:
    """Compute mean loss (and optionally accuracy) over a ``DataLoader``.

    Parameters are analogous to those in ``train_cnn``. When
    ``classification=True`` the function also computes accuracy.
    """
    model.eval()
    total_loss, total_samples = 0.0, 0
    correct = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if classification:
            if binary:
                yb = yb.float().view(-1, 1)   # <-- add this
            else:
                yb = yb.long().view(-1)       # <-- and this

        preds = model(xb)
        loss = criterion(preds, yb)
        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        if classification:
            if binary:
                probs = torch.sigmoid(preds.squeeze(1))
                preds_cls = (probs >= 0.5).long()
                correct += (preds_cls == yb.long().view(-1)).sum().item()
            else:
                preds_cls = preds.argmax(dim=1)
                correct += (preds_cls == yb).sum().item()

    out = {"loss": total_loss / max(total_samples, 1)}
    if classification and total_samples:
        out["accuracy"] = correct / total_samples
    return out



def _as_bct(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure CNN input is (B, C, T).
    Accepts (B, C, T) or (B, T, C) and permutes if needed.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor for CNN (B,C,T) or (B,T,C), got {tuple(x.shape)}")
    B, A, B2 = x.shape
    # Heuristic: channels should be "joint features" (e.g., 36), time should be long (>= 8)
    # If the middle dim looks like time (>=8) and the last looks like channels (<256), treat as (B,T,C)
    if A >= 8 and B2 < 256:
        return x.permute(0, 2, 1).contiguous()  # (B,T,C) -> (B,C,T)
    return x  # assume already (B,C,T)

def _prep_targets(y: torch.Tensor, *, classification: bool, binary: bool) -> torch.Tensor:
    """
    Make targets match the criterion expectations:
      - binary: float {0,1} with shape (B,1)
      - multi : long  class indices with shape (B,)
    """
    if not classification:
        return y  # not used here, but keep for completeness
    if binary:
        if y.dtype != torch.float32:
            y = y.float()
        return y.view(-1, 1)
    else:
        if y.dtype != torch.long:
            y = y.long()
        return y.view(-1)

@torch.no_grad()
def _evaluate_cnn(model: nn.Module, loader, criterion, device,
                  *, classification: bool, binary: bool) -> Dict[str, float]:
    model.eval()
    tot_loss, tot_n, correct = 0.0, 0, 0
    for xb, yb in loader:
        xb = _as_bct(xb.to(device))
        if classification:
            yb = _prep_targets(yb.to(device), classification=True, binary=binary)
        else:
            yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)
        bs = xb.size(0)
        tot_loss += loss.item() * bs
        tot_n += bs

        if classification:
            if binary:
                preds = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long()
                gold  = yb.long().view(-1)
            else:
                preds = logits.argmax(dim=1)
                gold  = yb
            correct += (preds == gold).sum().item()

    out = {"loss": tot_loss / max(tot_n, 1)}
    if classification and tot_n:
        out["accuracy"] = correct / tot_n
    return out

def train_cnn(model: nn.Module,
              train_loader,
              val_loader=None,
              *,
              epochs: int = 100,
              lr: float = 1e-3,
              weight_decay: float = 0.0,
              classification: bool = False,
              binary: bool = False,
              criterion: Optional[nn.Module] = None,
              optimizer_cls = torch.optim.Adam,
              device: Optional[torch.device] = None) -> Dict[str, float]:

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Choose criterion if not supplied
    if criterion is None:
        if classification:
            criterion = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

    opt = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None
    result: Dict[str, float] = {}

    for ep in range(1, epochs + 1):
        model.train()
        run_loss, run_n, run_correct = 0.0, 0, 0

        for xb, yb in train_loader:
            xb = _as_bct(xb.to(device))
            if classification:
                yb = _prep_targets(yb.to(device), classification=True, binary=binary)
            else:
                yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

            bs = xb.size(0)
            run_loss += loss.item() * bs
            run_n += bs

            if classification:
                if binary:
                    preds = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long()
                    gold  = yb.long().view(-1)
                else:
                    preds = logits.argmax(dim=1)
                    gold  = yb
                run_correct += (preds == gold).sum().item()

        train_loss = run_loss / max(run_n, 1)
        result["train_loss"] = train_loss
        if classification:
            result["train_accuracy"] = run_correct / max(run_n, 1)

        # validation
        if val_loader is not None:
            val_stats = _evaluate_cnn(model, val_loader, criterion, device,
                                      classification=classification, binary=binary)
            val_loss = val_stats["loss"]
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            result["val_loss"] = val_loss
            if classification:
                result["val_accuracy"] = val_stats.get("accuracy", 0.0)

    if best_state is not None:
        model.load_state_dict(best_state)

    if val_loader is None:
        result.setdefault("val_loss", result["train_loss"])
        if classification:
            result.setdefault("val_accuracy", result.get("train_accuracy", 0.0))

    return result