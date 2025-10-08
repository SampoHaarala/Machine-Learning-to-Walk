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
        preds = model(xb)
        loss = criterion(preds, yb)
        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        if classification:
            if binary:
                probs = torch.sigmoid(preds.squeeze(1))
                preds_cls = (probs >= 0.5).long()
                correct += (preds_cls == yb.long()).sum().item()
            else:
                preds_cls = preds.argmax(dim=1)
                correct += (preds_cls == yb).sum().item()

    result: Dict[str, float] = {"loss": total_loss / max(total_samples, 1)}
    if classification and total_samples > 0:
        result["accuracy"] = correct / total_samples
    return result


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
    classification: bool = False,
    binary: bool = False,
) -> Dict[str, float]:
    """Minimal training loop for the CNN supporting regression and classification.

    Parameters are similar to those of ``train_mlp``. When
    ``classification=True`` the default loss switches to
    ``BCEWithLogitsLoss`` (binary) or ``CrossEntropyLoss`` (multi‑class) and
    accuracies are computed. For regression the default remains ``MSELoss``.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Select criterion if none provided
    if criterion is None:
        if classification:
            criterion = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

    opt = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    step = 0

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if classification:
                if binary:
                    yb = yb.float().view(-1, 1)
                else:
                    yb = yb.long().view(-1)

            opt.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            bs = xb.size(0)
            running_loss += loss.item() * bs
            count += bs
            step += 1
            if classification:
                if binary:
                    probs = torch.sigmoid(preds.detach().squeeze(1))
                    preds_cls = (probs >= 0.5).long()
                    running_correct += (preds_cls == yb.long()).sum().item()
                else:
                    preds_cls = preds.detach().argmax(dim=1)
                    running_correct += (preds_cls == yb).sum().item()
            if log_interval and step % log_interval == 0:
                avg_loss = running_loss / max(count, 1)
                if classification:
                    avg_acc = running_correct / max(count, 1)
                    print(f"[ep {ep:03d} step {step:06d}] train_loss={avg_loss:.4f} train_acc={avg_acc:.4f}")
                else:
                    print(f"[ep {ep:03d} step {step:06d}] train_loss={avg_loss:.4f}")

        # End of epoch stats
        train_stats: Dict[str, float] = {"train_loss": running_loss / max(count, 1)}
        if classification:
            train_stats["train_accuracy"] = running_correct / max(count, 1)

        if val_loader is not None:
            val_result = _evaluate(model, val_loader, criterion, device, classification=classification, binary=binary)
            val_loss = val_result["loss"]
            train_stats["val_loss"] = val_loss
            if classification:
                train_stats["val_accuracy"] = val_result.get("accuracy", 0.0)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Epoch summary
        if classification:
            if val_loader is not None:
                print(f"[ep {ep:03d}] train_loss={train_stats['train_loss']:.4f} train_acc={train_stats['train_accuracy']:.4f} "
                      f"val_loss={train_stats['val_loss']:.4f} val_acc={train_stats['val_accuracy']:.4f}")
            else:
                print(f"[ep {ep:03d}] train_loss={train_stats['train_loss']:.4f} train_acc={train_stats['train_accuracy']:.4f}")
        else:
            if val_loader is not None:
                print(f"[ep {ep:03d}] train_loss={train_stats['train_loss']:.4f} val_loss={train_stats['val_loss']:.4f}")
            else:
                print(f"[ep {ep:03d}] train_loss={train_stats['train_loss']:.4f}")

    # Restore best validation weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Prepare final results
    result: Dict[str, float] = {}
    if classification:
        result["train_loss"] = train_stats["train_loss"]
        result["train_accuracy"] = train_stats["train_accuracy"]
        if val_loader is not None:
            result["val_loss"] = best_val_loss
            val_res = _evaluate(model, val_loader, criterion, device, classification=True, binary=binary)
            result["val_accuracy"] = val_res.get("accuracy", 0.0)
        else:
            result["val_loss"] = train_stats["train_loss"]
            result["val_accuracy"] = train_stats["train_accuracy"]
    else:
        result["train_loss"] = train_stats["train_loss"]
        result["val_loss"] = best_val_loss if val_loader is not None else train_stats["train_loss"]

    return result