from typing import List, Tuple
import json
import numpy as np
import torch
import torch.nn as nn

def build_mlp(input_dim: int, output_dim: int, hidden_layers: List[int] = None):
    """
    Construct a simple multi‑layer perceptron (MLP) using PyTorch.

    Parameters
    ----------
    input_dim : int
        The number of input features (e.g. number of joints * 3 for x/y/z).
    output_dim : int
        The dimension of the network's output. For regression tasks this is
        typically the same as input_dim; for classification tasks this might be
        the number of classes.
    hidden_layers : List[int], optional
        A list specifying the number of neurons in each hidden layer. If None
        is provided, a default configuration will be used.

    Returns
    -------
    nn.Module
        An untrained PyTorch neural network model.

    Notes
    -----
    This function only constructs the network architecture. You'll need to
    implement your own training loop to optimize it with your data.
    """
    if torch is None or nn is None:
        raise ImportError("PyTorch is required to build the MLP model.")

    if hidden_layers is None:
        # Provide a simple default: two hidden layers
        hidden_layers = [128, 64]

    layers = []
    in_features = input_dim
    for hidden_size in hidden_layers:
        layers.append(nn.Linear(in_features, hidden_size))
        layers.append(nn.ReLU())
        in_features = hidden_size

    layers.append(nn.Linear(in_features, output_dim))

    model = nn.Sequential(*layers)
    return model