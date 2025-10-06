from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import json
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None  # salli import ilman torchia

# --------- Esikäsittelyn asetukset ---------
@dataclass
class Preproc:
    rep: str = "rad"       # "deg" | "rad" | "sin_cos"
    unwrap: bool = True    # poista 360↔0 hyppäykset (vain deg/rad)
    center: bool = False   # z-score normalisointi treenissä (MLP)
    seq_len: int = 8       # CNN-ikkuna
    stride: int = 1        # CNN-liukuikkunan askel

# --------- Lataus: (T, 12) ---------
def load_rotations_12dof(path: str, *, rep: str = "rad", unwrap: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lukee JSONin jossa: {"frames":[ {"time":..., "rotations":[12 floattia]}, ... ]}
    Palauttaa:
      feats: (T,12) valituilla DoF:illa
      times: (T,)
    Esikäsittely:
      rep="deg"  -> asteet sellaisenaan
      rep="rad"  -> asteet -> rad, optional unwrap
      rep="sin_cos" -> jokainen kulma -> (sin,cos), jolloin ulottuvuus 20
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    frames = obj.get("frames") if isinstance(obj, dict) else obj
    if not isinstance(frames, list) or len(frames) == 0:
        raise ValueError("No frames found in JSON.")

    times = np.array([float(fr.get("time", 0.0)) for fr in frames], dtype=np.float32)
    R = np.array([fr["rotations"] for fr in frames], dtype=np.float32)  # (T,12)

    if rep == "deg":
        X = R
        if unwrap:
            # unwrap asteina: muunna rad -> unwrap -> takaisin deg
            rad = np.deg2rad(X)
            rad = np.unwrap(rad, axis=0)
            X = np.rad2deg(rad)
        return X.astype(np.float32), times

    if rep == "rad":
        rad = np.deg2rad(R)
        if unwrap:
            rad = np.unwrap(rad, axis=0)
        return rad.astype(np.float32), times

    if rep == "sin_cos":
        rad = np.deg2rad(R)
        # ei unwrapia, koska sin/cos poistaa diskontinuitetin
        X = np.concatenate([np.sin(rad), np.cos(rad)], axis=1)  # (T,20)
        return X.astype(np.float32), times

    raise ValueError(f"Unknown rep: {rep}")

# --------- Oppiparit ---------
def make_xy_mlp(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(T,D) -> X:(T-1,D), y:(T-1,D)"""
    return feats[:-1].astype(np.float32), feats[1:].astype(np.float32)

def make_xy_cnn(feats: np.ndarray, seq_len: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    (T,D) -> X:(N,D,L) kanavat-ensin (12/L tai 20/L sincos), y:(N,D)
    """
    T, D = feats.shape
    if T <= seq_len:
        raise ValueError(f"Need at least seq_len+1 frames (T={T}, L={seq_len}).")
    Xs, Ys = [], []
    for s in range(0, T - seq_len, stride):
        e = s + seq_len
        Xs.append(feats[s:e].T)   # (D,L)
        Ys.append(feats[e])       # (D,)
    X = np.stack(Xs, 0).astype(np.float32)
    y = np.stack(Ys, 0).astype(np.float32)
    return X, y

# --------- Torch Dataset/Loader ---------
class NumpyDataset(Dataset):  # type: ignore[misc]
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if torch is None:
            raise ImportError("PyTorch needed for Dataset/DataLoader.")
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self) -> int: return self.X.shape[0]
    def __getitem__(self, i: int): return self.X[i], self.y[i]

def split_train_val(X: np.ndarray, y: np.ndarray, split: float = 0.8):
    n = X.shape[0]
    n_tr = int(n * split)
    return (X[:n_tr], y[:n_tr]), (None if n_tr >= n else X[n_tr:], None if n_tr >= n else y[n_tr:])

def make_loaders_mlp(
    feats: np.ndarray, *, batch_size: int = 256, split: float = 0.8, center: bool = False
):
    X, y = make_xy_mlp(feats)  # (T-1,D)
    (Xtr, ytr), (Xva, yva) = split_train_val(X, y, split)

    if center:
        mu = Xtr.mean(0, keepdims=True); sigma = Xtr.std(0, keepdims=True) + 1e-8
        Xtr = (Xtr - mu) / sigma; ytr = (ytr - mu) / sigma
        if Xva is not None:
            Xva = (Xva - mu) / sigma; yva = (yva - mu) / sigma

    if torch is None:
        raise ImportError("PyTorch needed for DataLoader.")
    tr = DataLoader(NumpyDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    va = None if Xva is None else DataLoader(NumpyDataset(Xva, yva), batch_size=batch_size, shuffle=False)
    return tr, va

def make_loaders_cnn(
    feats: np.ndarray, *, seq_len: int = 8, stride: int = 1, batch_size: int = 256, split: float = 0.8
):
    X, y = make_xy_cnn(feats, seq_len=seq_len, stride=stride)  # X:(N,D,L), y:(N,D)
    if torch is None:
        raise ImportError("PyTorch needed for DataLoader.")
    (Xtr, ytr), (Xva, yva) = split_train_val(X, y, split)
    tr = DataLoader(NumpyDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    va = None if Xva is None else DataLoader(NumpyDataset(Xva, yva), batch_size=batch_size, shuffle=False)
    return tr, va

# --------- Päärajapinta: yksi funktio per käyttötapa ---------
def load_for_mlp(path: str, cfg: Preproc = Preproc()) -> Tuple["DataLoader", Optional["DataLoader"]]:
    feats, _ = load_rotations_12dof(path, rep=cfg.rep, unwrap=cfg.unwrap)
    return make_loaders_mlp(feats, batch_size=256, split=0.8, center=cfg.center)

def load_for_cnn(path: str, cfg: Preproc = Preproc()) -> Tuple["DataLoader", Optional["DataLoader"]]:
    feats, _ = load_rotations_12dof(path, rep=cfg.rep, unwrap=cfg.unwrap)
    return make_loaders_cnn(feats, seq_len=cfg.seq_len, stride=cfg.stride, batch_size=256, split=0.8)
