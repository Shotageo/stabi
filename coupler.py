# coupler.py — Bishop(simplified) + reinforcement (Tt)
from __future__ import annotations
import numpy as np, math

DEG = math.pi/180.0

def bishop_with_reinforcement(slices: dict, soil_top, Tt_per_slice: np.ndarray):
    """
    slices: compute_slices_poly_multi() が返す dict
    soil_top: Soil(dataclass)  ※単層想定（まずはここから）
    Tt_per_slice: 各スライスの弧接線方向の安定化力 [kN/m]
    """
    x = slices
    W = x["W"]; alpha = x["alpha"]; b = x["b"]
    Tt = np.asarray(Tt_per_slice, dtype=float)
    if Tt.shape != W.shape:
        Tt = np.resize(Tt, W.shape)

    c = float(soil_top.c); phi = float(soil_top.phi)
    tanp = math.tan(phi * DEG)

    Fs = 1.30
    for _ in range(120):
        denom_a = 1.0 + (tanp*np.tan(alpha)) / max(Fs, 1e-12)
        num = np.sum((c*b + W*tanp*np.cos(alpha) + Tt) / denom_a)
        den = np.sum(W*np.sin(alpha))
        if den <= 0: return None
        Fs_new = num/den
        if not (np.isfinite(Fs_new) and Fs_new > 0): return None
        if abs(Fs_new - Fs) < 1e-6:
            return float(Fs_new)
        Fs = Fs_new
    return float(Fs)
