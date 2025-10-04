import os, numpy as np, matplotlib
matplotlib.use("Agg")  # ← 追加：CIのヘッドレス環境で描画
import matplotlib.pyplot as plt
import stabi_lem as lem

import os, numpy as np, matplotlib.pyplot as plt
import stabi_lem as lem

os.makedirs("artifacts", exist_ok=True)

H,L = 25.0, 60.0
ground = lem.make_ground_example(H, L)
soils = [lem.Soil(18.0, 5.0, 30.0)]
interfaces = []
allow = []

xc, yc, R = 50.0, 40.0, 35.0
s = lem.arc_sample_poly_best_pair(ground, xc, yc, R, n=401, y_floor=0.0)
if s is not None:
    x1, x2, xs, ys, _h = s
    fig, ax = plt.subplots(figsize=(8,5))
    Xd = np.linspace(ground.X[0], ground.X[-1], 600)
    Yg = np.array([float(ground.y_at(x)) for x in Xd])
    ax.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Layer1")
    ax.plot(ground.X, ground.Y, lw=2.0, label="Ground")
    ax.plot(xs, ys, "r-", lw=2.0, label="Slip arc")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True); ax.legend()
    fig.savefig("artifacts/cross_section.png", dpi=140)
    plt.close(fig)

