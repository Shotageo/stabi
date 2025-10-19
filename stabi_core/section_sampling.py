from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import csv
import math
import numpy as np


def _read_csv(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return rows


def _has_cols(rows, cols: List[str]) -> bool:
    if not rows:
        return False
    r0 = {k.strip().lower() for k in rows[0].keys()}
    return all(c in r0 for c in [c.lower() for c in cols])


def build_section_profile_from_csv(
    csv_path: str,
    station: float,
    foot_xy: Tuple[float, float],
    nvec_xy: Tuple[float, float],
    u_span: float = 60.0,
    u_step: float = 1.0,
    station_tol: float = 0.51
) -> Dict[str, List[float]]:
    """
    CSVから横断プロファイル (u,z) を生成する。
    - station,offset,elevation があればその station を抽出
    - そうでなければ x,y,z を法線上に投影し、IDW補間で z(u)

    Returns: {"u": [...], "z": [...]}  （u: 法線方向オフセット, z: 標高）
    """
    rows = _read_csv(csv_path)
    if not rows:
        raise ValueError("CSVが空です。")

    # 1) station,offset,elevation 形式
    if _has_cols(rows, ["station", "offset", "elevation"]):
        u, z = [], []
        for r in rows:
            try:
                st = float(r["station"])
                if abs(st - station) <= station_tol:
                    u.append(float(r["offset"]))
                    z.append(float(r["elevation"]))
            except Exception:
                continue
        if not u:
            raise ValueError(f"CSV内に station≈{station:.2f} の横断が見つかりません。")
        ord_idx = np.argsort(u)
        return {"u": list(np.array(u)[ord_idx]), "z": list(np.array(z)[ord_idx])}

    # 2) x,y,z 形式（点群/測点）
    if not _has_cols(rows, ["x", "y", "z"]):
        raise ValueError("CSVに必要な列がありません。station/offset/elevation か x/y/z を用意してください。")

    px, py = foot_xy
    nx, ny = nvec_xy
    # 法線上のサンプル u
    u_samples = np.arange(-u_span, u_span + 1e-9, u_step)
    z_samples = np.zeros_like(u_samples, dtype=float)

    # 各点を法線に投影
    pts = []
    for r in rows:
        try:
            x, y, z = float(r["x"]), float(r["y"]), float(r["z"])
        except Exception:
            continue
        # ベクトル v = P - foot
        vx, vy = x - px, y - py
        u = vx * nx + vy * ny   # 法線方向成分
        d_perp = abs(vx * (-ny) + vy * nx)  # 法線に直交する成分の大きさ（簡易）
        pts.append((u, float(z), d_perp))
    if not pts:
        raise ValueError("x,y,z の有効点が見つかりません。")

    # IDWで z(u) を推定（近傍点のみ使用）
    pts = np.array(pts)  # [u, z, d_perp]
    for i, u in enumerate(u_samples):
        # u方向の近い点を選ぶ（±u_step*3）＆ 直交距離が小さい点を優先
        du = np.abs(pts[:, 0] - u)
        mask = du <= (u_step * 3.0)
        cand = pts[mask]
        if cand.size == 0:
            cand = pts[du.argmin()][None, :]
        # 重み： u差 と 直交距離 の合成（小さいほど重い）
        w = 1.0 / (1e-6 + du[mask] + 0.2 * cand[:, 2])
        z_est = float(np.sum(w * cand[:, 1]) / np.sum(w))
        z_samples[i] = z_est

    return {"u": list(u_samples), "z": list(z_samples)}

