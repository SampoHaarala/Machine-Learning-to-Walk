"""
model_server.py
================

This module provides a minimal TCP server that exposes a trained
neural network (MLP or CNN) for online pose prediction.  It is
compatible with the Unity ``SNNControllerModified`` script provided in
this repository.  The server accepts a comma‑separated list of
floating‑point joint rotations (in degrees) terminated by a newline,
predicts the next joint rotation vector and returns another
comma‑separated list of degrees.

The server performs a simple handshake on startup to communicate
simulation parameters.  It first sends a line ``"0,0,dt_ms"`` where
``dt_ms`` is the timestep in milliseconds (configured via ``--dt``).
It then waits for the client to send its own handshake (which is
ignored) before entering the prediction loop.

Both MLP and CNN models are supported.  For CNNs the server maintains
a sliding window of the last ``seq_len`` frames; once the window is
full it uses the CNN to predict the next frame given the stack of
frames.  Until the window is full the server returns the last received
input unchanged.

Usage
-----

::

    python model_server.py --weights trained_mlp.pt --model mlp --port 6900

    python model_server.py --weights trained_cnn.pt --model cnn \
        --seq-len 8 --dt 0.02 --port 6900

Note: For CNN you must specify ``--seq-len`` to match the value used
during training.
"""

from __future__ import annotations

import argparse
import socket
import threading
from typing import List, Optional, Tuple
from collections import deque

import numpy as np
import torch

from mlp_model import build_mlp
from cnn_model import build_cnn


def infer_dimension_from_weights(state: dict, model_type: str) -> int:
    """Infer input/output dimensionality D from a saved state dict.

    For MLP the first layer weight has shape (hidden_dim, D).  For CNN the
    first conv weight has shape (64, D, kernel).
    """
    if model_type == "mlp":
        # Look for first linear layer weight
        for k, v in state.items():
            if k.endswith("0.weight") and v.ndim == 2:
                return v.shape[1]
    else:
        # CNN: conv1.weight shape (out_channels=64, in_channels=D, k)
        for k, v in state.items():
            if "conv1.weight" in k and v.ndim == 3:
                return v.shape[1]
    raise ValueError("Could not infer input dimension from weights.")


class PoseServer:
    def __init__(self, weights: str, model_type: str, hidden: list[int], seq_len: int = 8,
                 dt: float = 0.02, device: Optional[str] = None) -> None:
        self.model_type = model_type
        self.seq_len = seq_len
        self.dt_ms = int(dt * 1000)
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # Load state dict
        state = torch.load(weights, map_location=self.device)
        D = infer_dimension_from_weights(state, model_type)
        # Build the appropriate architecture and load weights
        if model_type == "mlp":
            model = build_mlp(input_dim=D, hidden_layers=hidden, output_dim=D)
        else:
            model = build_cnn(input_channels=D, output_dim=D)
        
            # warm-up forward to create _fc with the correct flattened size
            with torch.no_grad():
                dummy = torch.zeros(1, D, self.seq_len, device=self.device)
                _ = model(dummy)

        model.load_state_dict(state)
        model.eval()
        self.model = model.to(self.device)
        self.D = D
        # Buffer for CNN window
        self.buffer: List[np.ndarray] = []

    def start(self, host: str = "0.0.0.0", port: int = 6900) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)  # may fail on Win
        except (AttributeError, OSError):
            pass
        srv.bind((host, port))
        srv.listen(1)
        print(f"[model_server] Listening on {host}:{port}")
        while True:
            conn, addr = srv.accept()
            print(f"[model_server] accept {addr}")
            threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()


    def _handle_client(self, conn: socket.socket, addr) -> None:
        """Read CSV angles (deg) line-by-line, run the model, return CSV angles."""
        conn_file = conn.makefile('rwb', buffering=0)
        try:
            # --- optional handshake back to Unity (rewards_size, patience_max, dt_ms)
            msg = f"0,0,{self.dt_ms}\n".encode("utf-8")
            conn_file.write(msg); conn_file.flush()

            buf = deque(maxlen=self.seq_len)  # for CNN windows
            D = None  # dimensionality inferred from first line

            while True:
                line = conn_file.readline()
                if not line:
                    break
                s = line.decode("utf-8").strip()
                if not s:
                    continue

                try:
                    obs = np.array([float(x) for x in s.split(",")], dtype=np.float32)
                except ValueError:
                    # bad line -> ignore
                    continue

                if D is None:
                    D = obs.shape[0]
                    # for CNN, prefill window with the first obs
                    if self.model_type == "cnn":
                        for _ in range(self.seq_len):
                            buf.append(obs.copy())

                # --- prepare input tensor
                if self.model_type == "mlp":
                    x = torch.from_numpy(obs).unsqueeze(0).to(self.device)   # (1, D)
                    with torch.no_grad():
                        y = self.model(x).cpu().numpy()[0]                   # (D,)
                else:  # cnn
                    buf.append(obs)
                    window = np.stack(list(buf), axis=0)                     # (L, D)
                    window = window.T[np.newaxis, ...].astype(np.float32)    # (1, D, L)
                    x = torch.from_numpy(window).to(self.device)
                    with torch.no_grad():
                        y = self.model(x).cpu().numpy()[0]                   # (D,)

                # --- send prediction back as CSV (deg)
                out = ",".join(f"{v:.6f}" for v in y) + "\n"
                print(out)
                conn_file.write(out.encode("utf-8")); conn_file.flush()

        except Exception as e:
            print(f"[model_server] Client error {addr}: {e}")
        finally:
            try: conn_file.close()
            except: pass
            try: conn.close()
            except: pass

    def _readline(self, conn: socket.socket) -> Optional[str]:
        """Read a line terminated by newline from a socket.  Returns None on EOF."""
        buffer = []
        while True:
            char = conn.recv(1)
            if not char:
                return None
            if char == b'\n':
                return b''.join(buffer).decode('ascii')
            buffer.append(char)

    def predict(self, feats_rad: np.ndarray) -> np.ndarray:
        """Run the model on the given feature vector (radians)."""
        if self.model_type == "mlp":
            x = torch.from_numpy(feats_rad).float().to(self.device)
            with torch.no_grad():
                y = self.model(x)
            return y.cpu().numpy()
        else:
            # Maintain a buffer of the last seq_len frames
            self.buffer.append(feats_rad)
            if len(self.buffer) < self.seq_len:
                # Pad buffer with copies of the first frame until we have enough
                return feats_rad.copy()
            elif len(self.buffer) > self.seq_len:
                # Drop oldest frame
                self.buffer = self.buffer[-self.seq_len:]
            window = np.stack(self.buffer, axis=1)  # (D,L)
            x = torch.from_numpy(window[None, ...]).float().to(self.device)  # (1,D,L)
            with torch.no_grad():
                y = self.model(x)
            return y.cpu().numpy()[0]
        
def parse_hidden(s: str):
    return [int(x) for x in s.split(",")] if s else []

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve a trained pose prediction model over TCP.")
    p.add_argument("--weights", required=True, help="Path to saved model (.pt).")
    p.add_argument("--model", choices=["mlp", "cnn"], required=True, help="Type of model to load.")
    p.add_argument("--hidden", type=str, default="256,128", help="Size of the hidden layer.")
    p.add_argument("--seq-len", type=int, default=8, help="Sequence length for CNN. Ignored for MLP.")
    p.add_argument("--dt", type=float, default=0.02, help="Simulation timestep in seconds.")
    p.add_argument("--port", type=int, default=6900, help="TCP port to listen on.")
    p.add_argument("--device", default=None, help="Force device (e.g. 'cpu' or 'cuda').")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    server = PoseServer(weights=args.weights, model_type=args.model, hidden=parse_hidden(args.hidden), 
                        seq_len=args.seq_len, dt=args.dt, device=args.device)
    server.start(port=args.port)


if __name__ == '__main__':
    main()