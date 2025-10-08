"""
Modified MLP model for both regression and classification.

This module exposes two main functions:

* ``build_mlp``: constructs a simple multi‐layer perceptron (MLP) with a
  configurable number of hidden layers. It supports both regression and
  classification tasks. When ``classification=True`` the ``output_dim``
  parameter is treated as the number of classes (use 1 for binary
  classification). A small boolean ``binary`` flag can be used to
  explicitly request a single output for two‑class problems; otherwise
  ``output_dim`` is used as the number of classes.

* ``train_mlp``: trains an ``nn.Module`` using a minimal training loop.
  It accepts a ``classification`` flag (and optional ``binary`` flag) to
  automatically select an appropriate loss function (``CrossEntropyLoss``
  for multi‑class or ``BCEWithLogitsLoss`` for binary classification).
  For regression the default remains ``MSELoss``. Targets are cast to
  ``torch.long`` for multi‑class classification. When a validation
  ``DataLoader`` is supplied the best model according to validation
  loss is restored at the end. The returned dictionary includes
  training/validation losses and optionally accuracies when performing
  classification.

The intent of these changes is to make the existing regression MLP
usable for animation classification without altering its basic API. A
``classification`` argument enables classification mode while the
``binary`` argument selects binary vs multi‑class losses. Users can
extend this to problems with more than two classes simply by setting
``classification=True`` and passing ``output_dim`` equal to the
number of categories.
"""

from __future__ import annotations

from typing import List, Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_layers: Optional[List[int]] = None,
    *,
    classification: bool = False,
    binary: bool = False,
) -> nn.Module:
    """Construct a fully connected network (MLP).

    Parameters
    ----------
    input_dim : int
        Number of input features (e.g., joints * features_per_joint).
    output_dim : int
        Size of network output. For classification problems this should
        equal the number of classes (use 1 for binary classification).
    hidden_layers : list[int] | None
        Sizes of hidden layers. If ``None`` a default two‑layer
        architecture [128, 64] is used.
    classification : bool, optional
        If ``True`` the network is intended for classification. This
        affects how the final layer is sized and how the training loop
        interprets the outputs. Defaults to ``False`` (regression).
    binary : bool, optional
        If ``classification=True`` and ``binary=True`` the final layer
        emits a single logit and training uses ``BCEWithLogitsLoss``.
        When ``classification=True`` and ``binary=False`` the final
        layer has ``output_dim`` units and training uses
        ``CrossEntropyLoss``. Ignored when ``classification=False``.

    Returns
    -------
    nn.Module
        A ``torch.nn`` module implementing the MLP.

    Notes
    -----
    Each hidden layer consists of ``Linear`` followed by ``ReLU``.
    No dropout or batch normalization is included. Activation on the
    output is deferred to the loss function (for classification) so that
    the training loop can use a numerically stable criterion.
    """
    # Default hidden configuration
    if hidden_layers is None:
        hidden_layers = [128, 64]

    layers: List[nn.Module] = []
    in_f = input_dim

    # Build hidden stack
    for h in hidden_layers:
        layers.append(nn.Linear(in_f, h))
        layers.append(nn.ReLU())
        in_f = h

    # Determine output dimension
    if classification and binary:
        out_dim = 1  # single logit for binary classification
    else:
        out_dim = output_dim

    # Final linear layer
    layers.append(nn.Linear(in_f, out_dim))

    return nn.Sequential(*layers)


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

    The model is set to evaluation mode and no gradients are computed.
    For classification tasks accuracy is also computed and returned.

    Parameters
    ----------
    model : nn.Module
        The network to evaluate.
    loader : DataLoader
        Provides (input, target) batches.
    criterion : nn.Module
        Loss function used to compute the average loss.
    device : torch.device
        Device on which evaluation is performed.
    classification : bool, optional
        Whether to compute accuracy as well as loss. Defaults to ``False``.
    binary : bool, optional
        Whether the classification task has two classes. Only used when
        ``classification=True``.

    Returns
    -------
    dict
        Dictionary with keys ``loss`` and optionally ``accuracy``.
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
                # For binary classification apply sigmoid to logits and
                # threshold at 0.5. yb should be float tensor with values in {0,1}.
                probs = torch.sigmoid(preds.squeeze(1))
                preds_cls = (probs >= 0.5).long()
                correct += (preds_cls == yb.long()).sum().item()
            else:
                # Multi‑class: use argmax over class dimension
                preds_cls = preds.argmax(dim=1)
                correct += (preds_cls == yb).sum().item()

    result: Dict[str, float] = {"loss": total_loss / max(total_samples, 1)}
    if classification and total_samples > 0:
        result["accuracy"] = correct / total_samples
    return result


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
    classification: bool = False,
    binary: bool = False,
) -> Dict[str, float]:
    """Train an MLP for regression or classification.

    Parameters
    ----------
    model : nn.Module
        The network produced by ``build_mlp``.
    train_loader : DataLoader
        Batches of (inputs, targets) for training.
    val_loader : DataLoader | None
        Optional validation data for tracking the best model.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate for the optimizer.
    weight_decay : float
        L2 regularization strength.
    criterion : nn.Module | None
        Loss function. If ``None``, a loss function is selected based
        on the ``classification`` and ``binary`` flags.
    optimizer_cls : callable
        Optimizer constructor (default Adam).
    device : torch.device | None
        Device on which to train. If ``None`` choose CUDA if available.
    grad_clip : float | None
        Maximum gradient norm for clipping. Disabled if ``None``.
    log_interval : int
        How often (in training steps) to print progress.
    classification : bool, optional
        Whether to treat the problem as classification. Defaults to ``False``.
    binary : bool, optional
        Whether the classification task is binary (only used if
        ``classification=True``).

    Returns
    -------
    dict
        Dictionary containing training/validation losses and, when
        classification is enabled, accuracies.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Select criterion if none provided
    if criterion is None:
        if classification:
            if binary:
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss()
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

            # Cast targets appropriately for classification
            if classification:
                if binary:
                    # Ensure targets are float tensor with shape (B,1)
                    yb = yb.float().view(-1, 1)
                else:
                    # Ensure integer class indices
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

            # Compute training accuracy on the fly
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

        # End of epoch metrics
        train_stats = {"train_loss": running_loss / max(count, 1)}
        if classification:
            train_stats["train_accuracy"] = running_correct / max(count, 1)

        # Validation
        if val_loader is not None:
            val_result = _evaluate(model, val_loader, criterion, device, classification=classification, binary=binary)
            val_loss = val_result["loss"]
            train_stats["val_loss"] = val_loss
            if classification:
                train_stats["val_accuracy"] = val_result.get("accuracy", 0.0)
            # Keep best weights on validation loss
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

    # Restore best validation weights if we have them
    if best_state is not None:
        model.load_state_dict(best_state)

    # Return final statistics (best validation if provided, else last train)
    result: Dict[str, float] = {}
    if classification:
        result["train_loss"] = train_stats["train_loss"]
        result["train_accuracy"] = train_stats["train_accuracy"]
        if val_loader is not None:
            result["val_loss"] = best_val_loss
            # Evaluate accuracy on best model
            val_res = _evaluate(model, val_loader, criterion, device, classification=True, binary=binary)
            result["val_accuracy"] = val_res.get("accuracy", 0.0)
        else:
            result["val_loss"] = train_stats["train_loss"]
            result["val_accuracy"] = train_stats["train_accuracy"]
    else:
        result["train_loss"] = train_stats["train_loss"]
        if val_loader is not None:
            result["val_loss"] = best_val_loss
        else:
            result["val_loss"] = train_stats["train_loss"]

    return result