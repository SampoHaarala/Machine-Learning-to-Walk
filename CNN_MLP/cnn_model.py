from typing import List, Tuple
import json
import numpy as np
import torch
import torch.nn as nn

def build_cnn(input_channels: int, output_dim: int, feature_size: Tuple[int, int] = (1, 1)):
    """
    Construct a simple convolutional neural network (CNN) using PyTorch.

    This stub is illustrative: depending on your task you may need a much
    deeper architecture or 1D convolutions instead of 2D. Here we use 1D
    convolutions on the temporal dimension of the joint data, treating joints
    as separate channels.

    Parameters
    ----------
    input_channels : int
        Number of channels in the input (e.g. joints * 3 features per joint).
    output_dim : int
        The dimension of the network's output (e.g. for regression or class count).
    feature_size : Tuple[int, int], optional
        The size to which features should be flattened before the final linear layer.

    Returns
    -------
    nn.Module
        An untrained PyTorch neural network model.

    Notes
    -----
    This model uses 1D convolutions along the temporal axis. Adjust kernel
    sizes, number of layers and pooling as needed for your data.
    """
    if torch is None or nn is None:
        raise ImportError("PyTorch is required to build the CNN model.")

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            # The input is expected to be of shape (batch_size, input_channels, seq_length)
            self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool1d(kernel_size=2)

            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool1d(kernel_size=2)

            # Compute the flattened feature size after convolutions and pooling. The provided
            # feature_size argument can override this if known.
            self.feature_size = feature_size
            flat_features = 128 * self.feature_size[0] * self.feature_size[1]
            self.fc = nn.Linear(flat_features, output_dim)

        def forward(self, x):
            # x shape: (batch, channels, seq_length)
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool2(x)
            # Flatten
            x = x.view(x.size(0), -1)
            return self.fc(x)

    return SimpleCNN()


__all__ = [
    "load_animation_json",
    "build_mlp",
    "build_cnn",
]