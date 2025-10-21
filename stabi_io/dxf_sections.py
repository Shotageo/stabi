# --- 追加: 地盤っぽさ判定ヘルパ ---
def _is_closed_poly(pts: np.ndarray, tol=1e-6) -> bool:
    if pts.shape[0] < 3: 
        return False
    return float(np.linalg.norm(pts[0] - pts[-1])) < tol

def _bbox(pts: np.ndarray):
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    return xmin, xmax, ymin, ymax

def _poly_length(pts: np.ndarray) -> float:
    if pts.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

def _rect_like_penalty(pts: np.ndarray) -> float:
    # 直交に近い折れ線ほどペナルティ（枠線対策）
    if pts.shape[0] < 4:
        return 0.0
    v = np.diff(pts, axis=0)
    v = v[np.linalg.norm(v, axis=1) > 1e-9]
    if len(v) < 2:
        return 0.0
    u = v[:-1] / np.linalg.norm(v[:-1], axis=1)[:, None]
    w = v[1:]  / np.linalg.norm(v[1:],  axis=1)[:, None]
    cosang = np.clip(np.sum(u*w, axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))  # 曲がり角
    # 90±10度が多いほど枠っぽい
    return float(np.mean((np.abs(ang-90.0) < 10.0).astype(float)))

def _groundlike_score(pts: np.ndarray) -> float:
    """
    横長・長い・頂点数多・非閉合・長方形っぽくない を高スコアに。
    0〜1 の範囲で返す（1が最良）。閾値は0.45目安。
    """
    if pts.shape[0] < 2:
        return 0.0
    L = _poly_length(pts)                             # 長さ
    xmin, xmax, ymin, ymax = _bbox(pts)
    w = max(1e-9, xmax - xmin)
    h = max(1e-9, ymax - ymin)
    aspect = h / w                                    # 横長ほど小さい
    nverts = pts.shape[0]
    closed = _is_closed_poly(pts, tol=1e-6)
    rectp = _rect_like_penalty(pts)

    # 正規化スコア
    s_len   = 1.0 - np.exp(-L/20.0)                   # 20mで ~0.63
    s_aspr  = np.clip(1.0 - aspect*2.0, 0.0, 1.0)     # aspect 0.0→1.0, 0.5→0
    s_nv    = np.clip((nverts-4)/20.0, 0.0, 1.0)      # 頂点多めが有利
    s_open  = 0.0 if closed else 1.0                  # 閉合は0
    s_rect  = 1.0 - np.clip(rectp, 0.0, 1.0)          # 直交だらけは減点

    # 重み付き合成
    s = 0.35*s_aspr + 0.25*s_len + 0.15*s_nv + 0.15*s_open + 0.10*s_rect
    return float(np.clip(s, 0.0, 1.0))
