"""
mlp_model.py
------------
Minimal, well-commented MLP builder + training loop.

This file provides:
- build_mlp(...)      -> returns a torch.nn.Module (the MLP)
- train_mlp(...)      -> trains the model with a simple loop
- _evaluate(...)      -> helper for validation loss

Assumptions
-----------
- Supervised learning with (inputs, targets) batches from a DataLoader.
- Default loss = MSE (regression). For classification, switch to CrossEntropyLoss.
"""

from __future__ import annotations

from typing import List, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_layers: Optional[List[int]] = None,
) -> nn.Module:
    """
    Construct a basic fully-connected network (MLP).

    Parameters
    ----------
    input_dim : int
        Number of input features (e.g., joints * features_per_joint).
    output_dim : int
        Size of network output (regression dim or #classes).
    hidden_layers : list[int] | None
        Sizes of hidden layers. If None, we use a sane default.

    Returns
    -------
    nn.Module
        The MLP as a torch Sequential model.

    Notes
    -----
    - Each hidden layer uses Linear -> ReLU.
    - No dropout/batchnorm by default (add if needed).
    """
    # Default two hidden layers if none provided
    if hidden_layers is None:
        hidden_layers = [256, 128]

    layers: List[nn.Module] = []
    in_f = input_dim

    # Build hidden stack: [Linear(in_f->h), ReLU()] for each hidden size
    for h in hidden_layers:
        layers.append(nn.Linear(in_f, h))
        layers.append(nn.ReLU())
        in_f = h

    # Final linear layer maps from last hidden width to output_dim
    layers.append(nn.Linear(in_f, output_dim))

    # Pack everything into a single Sequential module
    return nn.Sequential(*layers)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Compute mean loss over a DataLoader without gradient updates.

    - Sets model to eval() to disable dropout, etc.
    - Returns the average loss across all samples.
    """
    model.eval()  # evaluation mode (e.g., disables dropout)
    total, count = 0.0, 0

    for x, y in loader:
        # Move the batch to the selected device (CPU/GPU)
        x, y = x.to(device), y.to(device)

        # Forward pass -> predictions
        preds = model(x)

        # Compute per-batch loss (scalar tensor)
        loss = criterion(preds, y)

        # Accumulate sum of loss * batch_size for correct averaging
        bs = x.size(0)
        total += loss.item() * bs
        count += bs

    # Guard against empty loader (shouldn't happen, but safe)
    return total / max(count, 1)


def train_mlp(
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
    Minimal, readable training loop for MLPs.

    Parameters
    ----------
    model : nn.Module
        The MLP built by build_mlp (or any nn.Module).
    train_loader : DataLoader
        Yields (inputs, targets) batches for training.
    val_loader : DataLoader | None
        Optional validation data for early model selection.
    epochs : int
        Number of passes over the training set.
    lr : float
        Learning rate for the optimizer.
    weight_decay : float
        L2 regularization (0 disables).
    criterion : nn.Module | None
        Loss function. Default: nn.MSELoss().
        Use nn.CrossEntropyLoss() for classification (targets: LongTensor).
    optimizer_cls : callable
        Optimizer constructor (default Adam).
    device : torch.device | None
        If None: use "cuda" if available, else "cpu".
    grad_clip : float | None
        If set, clip global norm of gradients to this value.
    log_interval : int
        Print running train loss every N steps.

    Returns
    -------
    dict
        {"train_loss": last_epoch_train_loss,
         "val_loss": best_val_loss_or_last_train_if_no_val}
    """
    # Select device once; reuse every batch
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make sure model weights/buffers live on the same device as data
    model.to(device)

    # for sin/cos preprocessing (D = 2 * num_angles; e.g., 20 or 24)
    criterion = criterion or nn.MSELoss() # Mean Squared Error loss for regression
    #criterion = SineCosLoss(unit_penalty=1e-2, normalize_pred=True)
    #criterion = nn.CrossEntropyLoss() # Criterion for classification


    # Build optimizer from provided class (Adam by default)
    opt = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Track best validation loss to restore best weights later (early select)
    best_val = float("inf")
    best_state = None

    step = 0  # global step counter for logging
    for ep in range(1, epochs + 1):
        model.train()  # training mode (enables dropout, etc.)
        running = 0.0  # sum of (loss * batch_size) for this epoch
        count = 0      # number of samples seen this epoch

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # Reset gradients from previous step
            opt.zero_grad(set_to_none=True)

            # Forward pass
            preds = model(xb)

            # Compute loss for this batch
            
            loss = criterion(preds, yb)

            # Backpropagate to compute gradients
            loss.backward()

            # Optional gradient clipping to stabilize training
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Parameter update step
            opt.step()

            # Logging statistics
            bs = xb.size(0)
            running += loss.item() * bs
            count += bs
            step += 1

            # Periodic console logging of running average loss
            if log_interval and step % log_interval == 0:
                print(f"[ep {ep:03d} step {step:06d}] "
                      f"train_loss={running/max(count,1):.4f}")

        # Average training loss over the entire epoch
        train_loss = running / max(count, 1)

        # If a validation loader is provided, evaluate and track best model
        if val_loader is not None:
            val_loss = _evaluate(model, val_loader, criterion, device)

            # Keep a copy of parameters for the best validation loss
            if val_loss < best_val:
                best_val = val_loss
                # Clone tensors to CPU so we can safely restore later
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}

            print(f"[ep {ep:03d}] train={train_loss:.4f}  val={val_loss:.4f}")
        else:
            print(f"[ep {ep:03d}] train={train_loss:.4f}")

    # If we tracked a best validation checkpoint, restore it now
    if best_state is not None:
        model.load_state_dict(best_state)

    # Return final metrics for logging/plotting
    return {
        "train_loss": float(train_loss),
        "val_loss": float(best_val if val_loader is not None else train_loss),
    }


# -----------------------------------------------------------------------------
# Example usage (uncomment to try quickly)
# -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     # Fake data: 1000 samples, 32-dim inputs, 10-dim regression targets
#     x = torch.randn(1000, 32)
#     y = torch.randn(1000, 10)
#     ds = torch.utils.data.TensorDataset(x, y)
#     train_loader = DataLoader(ds, batch_size=64, shuffle=True)
#
#     model = build_mlp(input_dim=32, output_dim=10)
#     history = train_mlp(model, train_loader, epochs=5)
#     print(history)

class SineCosLoss(nn.Module):
    """
    pred,target: (B, 2*K) with layout [sin..., cos...]
    Optionally L2-normalizes predicted pairs to the unit circle.
    """
    def __init__(self, unit_penalty: float = 1e-2, normalize_pred: bool = True):
        super().__init__()
        self.unit_penalty = unit_penalty
        self.normalize_pred = normalize_pred

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, D = pred.shape
        assert D % 2 == 0, "Expect even dim: [sin..., cos...]"
        K = D // 2
        ps, pc = pred[:, :K], pred[:, K:]
        ts, tc = target[:, :K], target[:, K:]

        if self.normalize_pred:
            # project predicted pairs onto unit circle to avoid drift
            denom = torch.clamp(torch.sqrt(ps**2 + pc**2), min=1e-8)
            ps, pc = ps/denom, pc/denom

        # chordal MSE on unit circle
        mse = F.mse_loss(ps, ts) + F.mse_loss(pc, tc)

        # small penalty to keep (sin,cos) near unit circle even if normalize_pred=False
        unit_pen = ((ps**2 + pc**2 - 1.0)**2).mean()

        return mse + self.unit_penalty * unit_pen
