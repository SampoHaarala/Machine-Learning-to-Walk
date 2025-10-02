"""Visualisation tools for spiking neural networks.

This module contains a helper function for drawing the connectivity
structure of a spiking network with four rows
(input, hidden/cortex, cerebellum, output).  The visualisation
represents neurons as hollow circles arranged in rows.
Excitatory connections are drawn in green, inhibitory connections in red.
Active neurons can be highlighted by filling the circles.

Example:

    from snn_unity_project.python.snn_model import SpikingNetwork
    from snn_unity_project.python.visualizer import plot_network

    net = SpikingNetwork()
    fig = plot_network(net)
    fig.savefig('network.png')
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _get_attr(obj, name: str):
    """Return attribute if present, else None (to be robust across models)."""
    return getattr(obj, name, None)


def _layer_positions(n: int, y: float) -> np.ndarray:
    """Evenly distribute n x-coordinates in [0, 1] at fixed y."""
    if n <= 0:
        return np.array([], dtype=float)
    return np.linspace(0.05, 0.95, n)


def _draw_nodes(ax, x: np.ndarray, y: float,
                spikes: Optional[Sequence[float]],
                radius: float = 0.025) -> None:
    """Draw hollow nodes; fill if spikes indicator is provided and >0."""
    L = len(x)
    for i in range(L):
        face = 'white'
        if spikes is not None and i < len(spikes) and spikes[i] > 0.0:
            face = '#ffd700'  # bright yellow for active
        circ = plt.Circle((x[i], y), radius, edgecolor='black', facecolor=face, linewidth=1.0)
        ax.add_patch(circ)


def _draw_synapses(ax,
                   x_pre: np.ndarray, y_pre: float,
                   x_post: np.ndarray, y_post: float,
                   i_idx: np.ndarray, j_idx: np.ndarray, w: np.ndarray,
                   min_abs: float = 1e-6,
                   max_alpha: float = 0.8,
                   max_lw: float = 1.8) -> None:
    """Draw lines between pre and post according to indices and weights."""
    if w.size == 0:
        return
    absw = np.abs(w)
    wmax = absw.max() if absw.size > 0 else 0.0
    # Avoid division by zero; if all zero, skip
    if wmax <= 0.0:
        return

    # Normalize for alpha/width scaling
    norm = np.clip(absw / (wmax + 1e-12), 0.0, 1.0)
    for k in range(w.size):
        wij = w[k]
        if abs(wij) < min_abs:
            continue
        pre = i_idx[k]
        post = j_idx[k]
        if pre >= len(x_pre) or post >= len(x_post):
            continue
        color = 'green' if wij >= 0.0 else 'red'
        alpha = 0.2 + (max_alpha - 0.2) * norm[k]
        lw = 0.6 + (max_lw - 0.6) * norm[k]
        ax.plot([x_pre[pre], x_post[post]],
                [y_pre, y_post],
                color=color, alpha=alpha, linewidth=lw)


def plot_network(
    model: "SpikingNetwork",
    hidden_spikes: Optional[Sequence[float]] = None,
    cereb_spikes: Optional[Sequence[float]] = None,
    output_spikes: Optional[Sequence[float]] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Draw the network's structure in a simple grid layout (4 rows).

    Parameters
    ----------
    model : SpikingNetwork
        The network to visualise. Expected attributes:
        G_in, G_h, G_c, G_out (NeuronGroup) and any of:
        Sin (in→h), Shh (h→h), Shout (h→out),
        Sinc (in→c), Shc (h→c), Scout (c→out).
    hidden_spikes, cereb_spikes, output_spikes : optional sequences of float
        Binary indicators to highlight active neurons in each layer.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """

    # --- Determine layer sizes (be robust if some groups are missing) ---
    G_in = _get_attr(model, 'G_in')
    G_h  = _get_attr(model, 'G_h')
    G_c  = _get_attr(model, 'G_c')
    G_out= _get_attr(model, 'G_out')

    n_in  = int(G_in.N)  if G_in  is not None else 0
    n_h   = int(G_h.N)   if G_h   is not None else 0
    n_c   = int(G_c.N)   if G_c   is not None else 0
    n_out = int(G_out.N) if G_out is not None else 0

    # --- Layout (top to bottom) ---
    y_in, y_h, y_c, y_out = 3.0, 2.0, 1.0, 0.0
    x_in  = _layer_positions(n_in,  y_in)
    x_h   = _layer_positions(n_h,   y_h)
    x_c   = _layer_positions(n_c,   y_c)
    x_out = _layer_positions(n_out, y_out)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.axis('off')

    # --- Collect synapses present on the model (skip if missing) ---
    syn_list = []

    Sin   = _get_attr(model, 'Sin')
    if Sin is not None:
        syn_list.append(('in→h', Sin, x_in, y_in, x_h, y_h))

    Shh   = _get_attr(model, 'Shh')
    if Shh is not None:
        syn_list.append(('h→h', Shh, x_h, y_h,  x_h,  y_h))

    Shout = _get_attr(model, 'Shout')
    if Shout is not None:
        syn_list.append(('h→out', Shout, x_h, y_h, x_out, y_out))

    Sinc  = _get_attr(model, 'Sinc')
    if Sinc is not None:
        syn_list.append(('in→c', Sinc, x_in, y_in, x_c,  y_c))

    Shc   = _get_attr(model, 'Shc')
    if Shc is not None:
        syn_list.append(('h→c', Shc, x_h, y_h, x_c, y_c))

    Scout = _get_attr(model, 'Scout')
    if Scout is not None:
        syn_list.append(('c→out', Scout, x_c, y_c, x_out, y_out))

    # --- Draw each synapse set ---
    for tag, S, x_pre, y_pre, x_post, y_post in syn_list:
        try:
            # Brian2 arrays: one value per existing connection
            w = S.w[:]
            i = S.i[:]  # presyn indices
            j = S.j[:]  # postsyn indices
            _draw_synapses(ax, x_pre, y_pre, x_post, y_post, i, j, w)
        except Exception as ex:
            # If any synapse lacks .w/.i/.j, skip gracefully
            print(f"[visualizer] Skipping {tag}: {ex}")

    # --- Draw nodes last so they sit on top of lines ---
    _draw_nodes(ax, x_in, y_in, spikes=None)
    _draw_nodes(ax, x_h, y_h, spikes=hidden_spikes)
    _draw_nodes(ax, x_c, y_c, spikes=cereb_spikes)
    _draw_nodes(ax, x_out, y_out, spikes=output_spikes)

    # Optional: layer labels
    ax.text(0.01, y_in+0.12, "Input", fontsize=10, ha='left', va='bottom')
    ax.text(0.01, y_h+0.12, "Hidden/Ctx", fontsize=10, ha='left', va='bottom')
    ax.text(0.01, y_c+0.12, "Cerebellum", fontsize=10, ha='left', va='bottom')
    ax.text(0.01, y_out+0.12, "Output", fontsize=10, ha='left', va='bottom')

    return fig