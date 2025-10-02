# streamlit_app.py — 水位表示/編集対応（Dry/Offset/CSV/ru）＋監査UI
from __future__ import annotations
import streamlit as st
import numpy as np, heapq, time, hashlib, json, io, csv
import pandas as pd
import matplotlib.pyplot as plt

from stabi_lem import (
    Soil, GroundPL,
    make_ground_example, make_interface1_example, make_interface2_example,
    clip_interfaces_to_ground, arcs_from_center_by_entries_multi,
    fs_given_R_multi, arc_sample_poly_best_pair, driving_sum_for_R_multi,
)

st.set_page_config(page_title="Stabi LEM｜監査＆自動拡張", layout="wide")
st.title("Stabi LEM｜全センター監査 & 端ヒット自動拡張")

# ---------------- Quality ----------------
QUALITY = {
    "Coarse": dict(quick_slices=10, final_slices=30, n_entries_final=900,  probe_n_min_quick=81,
                   limit_arcs_quick=80,  show_k=60,  top_thick=10,
                   coarse_subsample="every 3rd", coarse_entries=160,
                   coarse_limit_arcs=50, coarse_probe_min=61,
                   budget_coarse_s=0.6, budget_quick_s=0.9,
                   audit_limit_per_center=10, audit_budget_s=2.0),
    "Normal": dict(quick_slices=12, final_slices=40, n_entries_final=1300, probe_n_min_quick=101,
                   limit_arcs_quick=120, show_k=120, top_thick=12,
                   coarse_subsample="every 2nd", coarse_entries=220,
                   coarse_limit_arcs=70, coarse_probe_min=81,
                   budget_coarse_s=0.8, budget_quick_s=1.2,
                   audit_limit_per_center=12, audit_budget_s=2.8),
    "Fine": dict(quick_slices=16, final_slices=50, n_entries_final=1700, probe_n_min_quick=121,
                 limit_arcs_quick=160, show_k=180, top_thick=16,
                 coarse_subsample="full", coarse_entries=320,
                 coarse_limit_arcs=100, coarse_probe_min=101,
                 budget_coarse_s=1.2, budget_quick_s=1.8,
                 audit_limit_per_center=16, audit_budget_s=3.2),
    "Very-fine": dict(quick_slices=20, final_slices=60, n_entries_final=2200, probe_n_min_quick=141,
                      limit_arcs_quick=220, show_k=240, top_thick=20,
                      coarse_subsample="full", coarse_entries=420,
                      coarse_limit_arcs=140, coarse_probe_min=121,
                      budget_coarse_s=1.8, budget_quick_s=2.6,
                      audit_limit_per_center=20, audit_budget_s=4.0),
}

# ---------------- Utils ----------------
def fs_to_color(fs: float):
    if fs < 1.0: return (0.85, 0.0, 0.0)
    if fs < 1.2:
        t = (fs - 1.0) / 0.2
        return (1.0, 0.50 + 0.50*t, 0.0)
    return (0.0, 0.55, 0.0)

def grid_points(x_min, x_max, y_min, y_max, nx, ny):
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    return [(float(xc), float(yc)) for yc in ys for xc in xs]

def hash_params(obj) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=float)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def near_edge(xc, yc, x_min, x_max, y_min, y_max, tol=1e-9):
    at_left  = abs(xc - x_min) < tol
    at_right = abs(xc - x_max) < tol
    at_bottom= abs(yc - y_min) < tol
    at_top   = abs(yc - y_max) < tol
    return at_left or at_right or at_bottom or at_top, dict(left=at_left,right=at_right,bottom=at_bottom,top=at_top)

def clip_yfloor(xs: np.ndarray, ys: np.ndarray, y_floor: float = 0.0):
    """描画安全化：y>=y_floor の区間だけ残す。2点未満なら None を返す。"""
    m = ys >= (y_floor - 1e-12)
    if np.count_nonzero(m) < 2:
        return None
    return xs[m], ys[m]

def build_linear_interpolator(xp: np.ndarray, yp: np.ndarray):
    xp = np.asarray(xp, dtype=float); yp = np.asarray(yp, dtype=float)
    order = np.argsort(xp); xp = xp[order]; yp = yp[order]
    def zfun(x):
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        out[x <= xp[0]] = yp[0]
        out[x >= xp[-1]] = yp[-1]
        mid = (x > xp[0]) & (x < xp[-1])
        out[mid] = np.interp(x[mid], xp, yp)
        return out
    return zfun

# ---------------- Inputs ----------------
with st.form("params"):
    A, B = st.columns(2)
    with A:
        st.subheader("Geometry")
        H = st.number_input("H (m)", 5.0, 200.0, 25.0, 0.5)
        L = st.number_input("L (m)", 5.0, 400.0, 60.0, 0.5)
        ground = make_ground_example(H, L)

        st.subheader("Layers")
        n_layers = st.selectbox("Number of layers", [1,2,3], index=2)
        interfaces = []
        if n_layers >= 2: interfaces.append(make_interface1_example(H, L))
        if n_layers >= 3: interfaces.append(make_interface2_example(H, L))

        st.subheader("Soils (top→bottom)")
        s1 = Soil(st.number_input("γ₁", 10.0, 25.0, 18.0, 0.5),
                  st.number_input("c₁", 0.0, 200.0, 5.0, 0.5),
                  st.number_input("φ₁", 0.0, 45.0, 30.0, 0.5))
        soils = [s1]
        if n_layers >= 2:
            s2 = Soil(st.number_input("γ₂", 10.0, 25.0, 19.0, 0.5),
                      st.number_input("c₂", 0.0, 200.0, 8.0, 0.5),
                      st.number_input("φ₂", 0.0, 45.0, 28.0, 0.5))
            soils.append(s2)
        if n_layers >= 3:
            s3 = Soil(st.number_input("γ₃", 10.0, 25.0, 20.0, 0.5),
                      st.number_input("c₃", 0.0, 200.0, 12.0, 0.5),
                      st.number_input("φ₃", 0.0, 45.0, 25.0, 0.5))
            soils.append(s3)

        st.subheader("Crossing control")
        allow_cross=[]
        if n_layers>=2: allow_cross.append(st.checkbox("Allow into Layer 2", True))
        if n_layers>=3: allow_cross.append(st.checkbox("Allow into Layer 3", True))

        st.subheader("Target safety")
        Fs_target = st.number_input("Target FS", 1.00, 2.00, 1.20, 0.05)

    with B:
        st.subheader("Center grid")
        x_min = st.number_input("x min", 0.20*L, 3.00*L, 0.25*L, 0.05*L)
        x_max = st.number_input("x max", 0.30*L, 4.00*L, 1.15*L, 0.05*L)
        y_min = st.number_input("y min", 0.80*H, 7.00*H, 1.60*H, 0.10*H)
        y_max = st.number_input("y max", 1.00*H, 8.00*H, 2.20*H, 0.10*H)
        nx = st.slider("nx", 6, 60, 14)
        ny = st.slider("ny", 4, 40, 9)

        st.subheader("Method / Quality")
        method = st.selectbox("Method", ["Bishop (simplified)","Fellenius"])
        quality = st.select_slider("Quality", options=list(QUALITY.keys()), value="Normal")
        with st.expander("Advanced", expanded=False):
            override = st.checkbox("Override Quality", value=False)
            quick_slices_in  = st.slider("Quick slices", 6, 40, QUALITY[quality]["quick_slices"], 1, disabled=not override)
            final_slices_in  = st.slider("Final slices", 20, 80, QUALITY[quality]["final_slices"], 2, disabled=not override)
            n_entries_final_in = st.slider("Final n_entries", 200, 4000, QUALITY[quality]["n_entries_final"], 100, disabled=not override)
            probe_min_q_in   = st.slider("Quick min probe", 41, 221, QUALITY[quality]["probe_n_min_quick"], 10, disabled=not override)
            limit_arcs_q_in  = st.slider("Quick max arcs/center", 20, 400, QUALITY[quality]["limit_arcs_quick"], 10, disabled=not override)
            budget_coarse_in = st.slider("Budget Coarse (s)", 0.1, 5.0, QUALITY[quality]["budget_coarse_s"], 0.1, disabled=not override)
            budget_quick_in  = st.slider("Budget Quick (s)", 0.1, 5.0, QUALITY[quality]["budget_quick_s"], 0.1, disabled=not override)

        st.subheader("Depth range (vertical)")
        depth_min = st.number_input("Depth min (m)", 0.0, 50.0, 0.5, 0.5)
        depth_max = st.number_input("Depth max (m)", 0.5, 50.0, 4.0, 0.5)

        # ==== Water UI ====
        st.subheader("Water")
        water_mode = st.radio("Water model", ["Dry", "Offset from ground", "CSV (x,z)", "ru model"],
                              index=0, horizontal=True)
        water = {"type":"dry"}
        editable_points_json = None  # パラメータキー用

        if water_mode == "Offset from ground":
            offset = st.number_input("Offset (m): +above / -below ground", -50.0, 50.0, -1.5, 0.1)
            # 初期点を地表+offset から作成し、表で編集
            n_init = st.slider("Initial control points", 4, 40, 12, 1)
            xp0 = np.linspace(float(0.0), float(L), n_init)
            yp0 = np.array([float(ground.y_at(x)) + float(offset) for x in xp0], dtype=float)
            df0 = pd.DataFrame({"x": xp0, "z": yp0})
            st.caption("Offset から初期化した水位線の制御点（行の追加/削除・x,zの修正が可能）")
            edited = st.data_editor(
                df0, use_container_width=True,
                num_rows="dynamic", key="wt_offset_editor",
                column_config={
                    "x": st.column_config.NumberColumn(format="%.3f", step=0.1, help="水平位置（地表xと同じ単位）"),
                    "z": st.column_config.NumberColumn(format="%.3f", step=0.1, help="標高")
                }
            )
            # 入力の正規化：x は [0, L] にクリップ
            if len(edited) >= 2:
                xp = np.clip(edited["x"].to_numpy(dtype=float), 0.0, float(L))
                yp = edited["z"].to_numpy(dtype=float)
                editable_points_json = edited.to_json()  # 再計算トリガ
                zfun = build_linear_interpolator(xp, yp)
                water = {"type":"WT", "z": zfun, "meta":{"source":"offset+edit", "n": int(len(xp))}}
            else:
                st.warning("水位線の制御点は2点以上必要です。")
                zfun = build_linear_interpolator(np.array([0.0, L]), np.array([yp0[0], yp0[-1]]))
                water = {"type":"WT", "z": zfun, "meta":{"source":"offset+edit(fallback)", "n": 2}}

        elif water_mode == "CSV (x,z)":
            up = st.file_uploader("Upload waterline CSV (columns: x,z)", type=["csv"])
            if up is not None:
                data = up.read().decode("utf-8")
                rd = csv.reader(io.StringIO(data))
                # ヘッダ自動判定
                try:
                    snif = csv.Sniffer()
                    if snif.has_header(data[:1024]): next(rd, None)
                except Exception:
                    pass
                xs, zs = [], []
                for row in rd:
                    if len(row) < 2: continue
                    try:
                        xs.append(float(row[0])); zs.append(float(row[1]))
                    except:
                        pass
                if len(xs) >= 2:
                    df0 = pd.DataFrame({"x": xs, "z": zs})
                    st.caption("CSV 読み込み後、必要なら制御点を編集してください（行の追加/削除・x,z修正可）")
                    edited = st.data_editor(
                        df0, use_container_width=True,
                        num_rows="dynamic", key="wt_csv_editor",
                        column_config={
                            "x": st.column_config.NumberColumn(format="%.3f", step=0.1),
                            "z": st.column_config.NumberColumn(format="%.3f", step=0.1)
                        }
                    )
                    if len(edited) >= 2:
                        xp = edited["x"].to_numpy(dtype=float)
                        yp = edited["z"].to_numpy(dtype=float)
                        editable_points_json = edited.to_json()
                        zfun = build_linear_interpolator(xp, yp)
                        water = {"type":"WT", "z": zfun, "meta":{"source":"csv+edit", "n": int(len(xp))}}
                    else:
                        st.warning("水位線の制御点は2点以上必要です。")
                        water = {"type":"dry"}
                else:
                    st.warning("CSV は少なくとも2点 (x,z) が必要です。")
                    water = {"type":"dry"}
            else:
                st.info("CSV をアップロードしてください。")

        elif water_mode == "ru model":
            # ← 要望どおり「水位線は描かない」。係数のみ。
            ru = st.slider("ru (0.0–1.0)", 0.0, 1.0, 0.30, 0.01,
                           help="底面有効法を N'=(1-ru)N と近似。一般に 0.1〜0.5 程度で感度検討。")
            water = {"type":"ru", "ru": float(ru)}

        else:
            water = {"type":"dry"}

    run = st.form_submit_button("▶ 計算開始")

# ---------------- Quality expand ----------------
P = QUALITY[quality].copy()
if 'override' in locals() and override:
    P.update(dict(
        quick_slices=quick_slices_in, final_slices=final_slices_in,
        n_entries_final=n_entries_final_in, probe_n_min_quick=probe_min_q_in,
        limit_arcs_quick=limit_arcs_q_in,
        budget_coarse_s=budget_coarse_in, budget_quick_s=budget_quick_in,
    ))

# ---------------- Keys ----------------
def param_pack():
    return dict(
        H=H, L=L, n_layers=n_layers,
        soils=[(s.gamma, s.c, s.phi) for s in soils],
        allow_cross=allow_cross, Fs_target=Fs_target,
        center=[x_min, x_max, y_min, y_max, nx, ny],
        method=method, quality=P, depth=[depth_min, depth_max],
        water_mode=water_mode,
        # 編集表がある場合は内容でキーを変えて再計算をトリガ
        editable_points=st.session_state.get("wt_offset_editor") or st.session_state.get("wt_csv_editor")
    )
param_key = hash_params(param_pack())

# ---------------- Compute ----------------
# ---------------- Compute ----------------
def compute_once():
    # === パラメータ ===
    EXT_FACTOR = 0.30   # 1回の拡張率（幅/高さの30%を外側へ足す）
    MAX_EXT_ROUNDS = 2  # 最大拡張ラウンド数
    FS_IMPROVE_TOL = 1e-4  # 改善がこの相対量未満なら打ち切り（0.01%）

    def subsampled_centers(xmin, xmax, ymin, ymax):
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        tag = P["coarse_subsample"]
        if tag == "every 3rd":
            xs = xs[::3] if len(xs)>2 else xs
            ys = ys[::3] if len(ys)>2 else ys
        elif tag == "every 2nd":
            xs = xs[::2] if len(xs)>1 else xs
            ys = ys[::2] if len(ys)>1 else ys
        return [(float(xc), float(yc)) for yc in ys for xc in xs]

    def pick_center(xmin, xmax, ymin, ymax, budget_s):
        deadline = time.time() + budget_s
        best = None; tested=[]
        for (xc,yc) in subsampled_centers(xmin, xmax, ymin, ymax):
            cnt=0; Fs_min=None
            for _x1,_x2,_R,Fs in arcs_from_center_by_entries_multi(
                ground, soils, xc, yc,
                n_entries=P["coarse_entries"], method="Fellenius",
                depth_min=depth_min, depth_max=depth_max,
                interfaces=interfaces, allow_cross=allow_cross,
                quick_mode=True, n_slices_quick=max(8, P["quick_slices"]//2),
                limit_arcs_per_center=P["coarse_limit_arcs"],
                probe_n_min=P["coarse_probe_min"],
                water=water,
            ):
                cnt+=1
                if (Fs_min is None) or (Fs < Fs_min): Fs_min = Fs
                if time.time() > deadline: break
            tested.append((xc,yc))
            score = (cnt, - (Fs_min if Fs_min is not None else 1e9))
            if (best is None) or (score > best[0]): best = (score, (xc,yc))
            if time.time() > deadline: break
        return (best[1] if best else None), tested

    def refine_at_center(xc, yc):
        # Quick: R 候補
        heap_R=[]; deadline=time.time()+P["budget_quick_s"]
        for _x1,_x2,R,Fs in arcs_from_center_by_entries_multi(
            ground, soils, xc, yc,
            n_entries=P["n_entries_final"], method="Fellenius",
            depth_min=depth_min, depth_max=depth_max,
            interfaces=interfaces, allow_cross=allow_cross,
            quick_mode=True, n_slices_quick=P["quick_slices"],
            limit_arcs_per_center=P["limit_arcs_quick"],
            probe_n_min=P["probe_n_min_quick"],
            water=water,
        ):
            heapq.heappush(heap_R, (-Fs, R))
            if len(heap_R) > max(P["show_k"], P["top_thick"] + 20): heapq.heappop(heap_R)
            if time.time() > deadline: break
        R_candidates = [r for _fsneg, r in sorted([(-fsneg,R) for fsneg,R in heap_R], key=lambda t:t[0])]
        if not R_candidates:
            return None, None

        # Refine
        refined_local=[]
        for R in R_candidates[:P["show_k"]]:
            Fs = fs_given_R_multi(ground, interfaces, soils, allow_cross, method, xc, yc, R,
                                  n_slices=P["final_slices"], water=water)
            if Fs is None: continue
            s = arc_sample_poly_best_pair(ground, xc, yc, R, n=251, y_floor=0.0)
            if s is None: continue
            x1,x2,*_ = s
            packD = driving_sum_for_R_multi(ground, interfaces, soils, allow_cross, xc, yc, R,
                                            n_slices=P["final_slices"], water=water)
            if packD is None: continue
            D_sum,_,_ = packD
            T_req = max(0.0, (Fs_target - Fs)*D_sum)
            refined_local.append(dict(Fs=float(Fs), R=float(R), x1=float(x1), x2=float(x2), T_req=float(T_req)))
        if not refined_local:
            return [], None
        refined_local.sort(key=lambda d:d["Fs"])
        return refined_local, float(refined_local[0]["Fs"])

    # ====== ここから自動拡張ループ ======
    # 初期グリッド
    cur = dict(xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_max)
    history = []
    best_pack = None  # (Fs_min, res_dict)

    for round_id in range(MAX_EXT_ROUNDS + 1):
        with st.spinner(("Coarse/Quick/Refine 実行中" if round_id==0 else f"再探索（拡張#{round_id}）")):
            center, tested = pick_center(cur["xmin"], cur["xmax"], cur["ymin"], cur["ymax"], P["budget_coarse_s"])
            if center is None:
                # これまでのベストがあればそれを返す
                if best_pack is not None:
                    break
                return dict(error="Coarseで候補なし。枠/深さを広げてください。")
            xc, yc = center
            refined, Fs_min = refine_at_center(xc, yc)
            if not refined:
                # ベストがあれば採用、なければエラー
                if best_pack is not None:
                    break
                return dict(error="Refineで有効弧なし。設定/Qualityを見直してください。")

        # 今回結果のまとめ
        idx_minFs = int(np.argmin([d["Fs"] for d in refined]))
        idx_maxT  = int(np.argmax([d["T_req"] for d in refined]))
        centers_disp  = grid_points(x_min, x_max, y_min, y_max, nx, ny)          # 表示用は元の範囲
        centers_audit = grid_points(cur["xmin"], cur["xmax"], cur["ymin"], cur["ymax"], nx, ny)  # 監査は拡張後範囲

        res_now = dict(center=(xc,yc), refined=refined,
                       idx_minFs=idx_minFs, idx_maxT=idx_maxT,
                       centers_disp=centers_disp, centers_audit=centers_audit)

        # ベスト更新
        if (best_pack is None) or (Fs_min < best_pack[0] - FS_IMPROVE_TOL*best_pack[0]):
            best_pack = (Fs_min, res_now)

        # 端ヒット判定（センターがグリッド外周に“貼り付いた”か）
        hit, where = near_edge(xc, yc, cur["xmin"], cur["xmax"], cur["ymin"], cur["ymax"])
        history.append(dict(round=round_id, grid=dict(**cur), center=(xc,yc), hit=where, Fs=Fs_min))

        # 拡張する？（条件：ヒット & ラウンド余裕）
        if (not hit) or (round_id >= MAX_EXT_ROUNDS):
            break

        # どの方向へ伸ばすか
        dx = (cur["xmax"] - cur["xmin"])
        dy = (cur["ymax"] - cur["ymin"])
        grew = False
        if where["left"]:
            cur["xmin"] = cur["xmin"] - EXT_FACTOR*dx; grew=True
        if where["right"]:
            cur["xmax"] = cur["xmax"] + EXT_FACTOR*dx; grew=True
        if where["bottom"]:
            cur["ymin"] = cur["ymin"] - EXT_FACTOR*dy; grew=True
        if where["top"]:
            cur["ymax"] = cur["ymax"] + EXT_FACTOR*dy; grew=True

        # 伸ばしたがさらに安全側に限る：下限など最小値の制約はお好みで
        # ここでは下方向の下限を 0.5*H に制限しない（深い円弧も探索するため）
        if not grew:
            break

    # ループ終了：ベストを採用
    Fs_best, res_best = best_pack
    # タイトル尾に履歴を短く付記
    note = []
    if len(history) > 1:
        last = history[-1]
        note.append(f"auto-extend {len(history)-1}x")
        sides=[]
        for k in ("left","right","bottom","top"):
            if any(h["hit"][k] for h in history):
                sides.append(k[0].upper())
        if sides:
            note.append("dirs=" + "".join(sides))
    res_best["expand_note"] = " / ".join(note) if note else None
    return res_best

    refined=[]
    for R in R_candidates[:P["show_k"]]:
        Fs = fs_given_R_multi(ground, interfaces, soils, allow_cross, method, xc, yc, R,
                              n_slices=P["final_slices"], water=water)
        if Fs is None: continue
        # span（描画用, 計算は stabi_lem 内で y_floor=-inf 前提）
        s = arc_sample_poly_best_pair(ground, xc, yc, R, n=251, y_floor=0.0)
        if s is None: continue
        x1,x2,*_ = s
        packD = driving_sum_for_R_multi(ground, interfaces, soils, allow_cross, xc, yc, R,
                                        n_slices=P["final_slices"], water=water)
        if packD is None: continue
        D_sum,_,_ = packD
        T_req = max(0.0, (Fs_target - Fs)*D_sum)
        refined.append(dict(Fs=float(Fs), R=float(R), x1=float(x1), x2=float(x2), T_req=float(T_req)))
    if not refined:
        return dict(error="Refineで有効弧なし。設定/Qualityを見直してください。")
    refined.sort(key=lambda d:d["Fs"])
    idx_minFs = int(np.argmin([d["Fs"] for d in refined]))
    idx_maxT  = int(np.argmax([d["T_req"] for d in refined]))

    centers_disp = grid_points(x_min, x_max, y_min, y_max, nx, ny)
    centers_audit= grid_points(x_min_a, x_max_a, y_min_a, y_max_a, nx, ny)

    return dict(center=(xc,yc), refined=refined,
                idx_minFs=idx_minFs, idx_maxT=idx_maxT,
                centers_disp=centers_disp, centers_audit=centers_audit,
                expand_note=expand_note, water_mode=water_mode, water=water)

# run
if run or ("last_key" not in st.session_state) or (st.session_state["last_key"] != param_key):
    res = compute_once()
    if "error" in res: st.error(res["error"]); st.stop()
    st.session_state["last_key"] = param_key
    st.session_state["res"] = res

res = st.session_state["res"]
xc,yc = res["center"]
refined = res["refined"]; idx_minFs = res["idx_minFs"]; idx_maxT=res["idx_maxT"]
centers_disp = res["centers_disp"]; centers_audit = res["centers_audit"]
water_mode = res.get("water_mode","Dry")
water = res.get("water", {"type":"dry"})

# ---------------- After-run toggles ----------------
st.subheader("表示オプション")
c1,c2,c3,c4 = st.columns([1,1,1,2])
with c1:
    show_centers = st.checkbox("Show center-grid (all points)", True)
with c2:
    show_all_refined = st.checkbox("Show refined arcs (Fs-colored)", True)
with c3:
    show_minFs = st.checkbox("Show Min Fs", True)
    show_maxT  = st.checkbox("Show Max required T", True)
with c4:
    audit_show = st.checkbox("Show arcs from ALL centers (Quick audit)", False)
    audit_limit = st.slider("Audit: max arcs/center", 5, 40, QUALITY[quality]["audit_limit_per_center"], 1, disabled=not audit_show)
    audit_budget = st.slider("Audit: total budget (sec)", 1.0, 6.0, QUALITY[quality]["audit_budget_s"], 0.1, disabled=not audit_show)
    audit_seed   = st.number_input("Audit seed", 0, 9999, 0, disabled=not audit_show)

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(10.5, 7.5))

Xd = np.linspace(0.0, float(ground.X[-1]), 600)
Yg = np.array([float(ground.y_at(float(x))) for x in Xd], dtype=float)

if n_layers==1:
    ax.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Layer1")
elif n_layers==2:
    Y1 = clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0]
    ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
    ax.fill_between(Xd, 0.0, Y1, alpha=0.12, label="Layer2")
else:
    Y1,Y2 = clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)
    ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
    ax.fill_between(Xd, Y2, Y1, alpha=0.12, label="Layer2")
    ax.fill_between(Xd, 0.0, Y2, alpha=0.12, label="Layer3")

# 地表・層
ax.plot(ground.X, ground.Y, linewidth=2.2, label="Ground")
if n_layers>=2:
    ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0], linestyle="--", linewidth=1.2, label="Interface 1")
if n_layers>=3:
    ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)[1], linestyle="--", linewidth=1.2, label="Interface 2")

# 水位線（WTのみ描画。ru/dry は描かない）
if water.get("type") == "WT":
    zfun = water["z"]
    Zw = zfun(Xd) if callable(zfun) else np.full_like(Xd, float(zfun))
    ax.plot(Xd, Zw, color="tab:blue", linewidth=1.6, alpha=0.85, label="Water table")

# 外周
ax.plot([ground.X[-1], ground.X[-1]],[0.0, ground.y_at(ground.X[-1])], linewidth=1.0)
ax.plot([ground.X[0],  ground.X[-1]],[0.0, 0.0],                       linewidth=1.0)
ax.plot([ground.X[0],  ground.X[0]], [0.0, ground.y_at(ground.X[0])],  linewidth=1.0)

# center-grid
if show_centers:
    xs=[c[0] for c in centers_disp]; ys=[c[1] for c in centers_disp]
    ax.scatter(xs, ys, s=12, c="k", alpha=0.25, marker=".", label="Center grid")
# chosen center
ax.scatter([xc],[yc], s=70, marker="s", color="tab:blue", label="Chosen center")

# refined（描画時クリップ）
if show_all_refined:
    for d in refined:
        xs=np.linspace(d["x1"], d["x2"], 200)
        ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
        clipped = clip_yfloor(xs, ys, y_floor=0.0)
        if clipped is None: 
            continue
        xs_c, ys_c = clipped
        ax.plot(xs_c, ys_c, linewidth=0.9, alpha=0.75, color=fs_to_color(d["Fs"]))

# pick-ups（描画時クリップ）
if show_minFs and 0<=idx_minFs<len(refined):
    d=refined[idx_minFs]
    xs=np.linspace(d["x1"], d["x2"], 400)
    ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
    clipped = clip_yfloor(xs, ys, y_floor=0.0)
    if clipped is not None:
        xs_c, ys_c = clipped
        ax.plot(xs_c, ys_c, linewidth=3.0, color=(0.9,0.0,0.0), label=f"Min Fs = {d['Fs']:.3f}")
        y1=float(ground.y_at(xs_c[0])); y2=float(ground.y_at(xs_c[-1]))
        ax.plot([xc,xs_c[0]],[yc,y1], linewidth=1.1, color=(0.9,0.0,0.0), alpha=0.9)
        ax.plot([xc,xs_c[-1]],[yc,y2], linewidth=1.1, color=(0.9,0.0,0.0), alpha=0.9)

if show_maxT and 0<=idx_maxT<len(refined):
    d=refined[idx_maxT]
    xs=np.linspace(d["x1"], d["x2"], 400)
    ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
    clipped = clip_yfloor(xs, ys, y_floor=0.0)
    if clipped is not None:
        xs_c, ys_c = clipped
        ax.plot(xs_c, ys_c, linewidth=3.0, linestyle="--", color=(0.2,0.0,0.8),
                label=f"Max required T = {d['T_req']:.1f} kN/m (Fs={d['Fs']:.3f})")
        y1=float(ground.y_at(xs_c[0])); y2=float(ground.y_at(xs_c[-1]))
        ax.plot([xc,xs_c[0]],[yc,y1], linewidth=1.1, linestyle="--", color=(0.2,0.0,0.8), alpha=0.9)
        ax.plot([xc,xs_c[-1]],[yc,y2], linewidth=1.1, linestyle="--", color=(0.2,0.0,0.8), alpha=0.9)

# axis & legend
x_upper = max(1.18*float(ground.X[-1]), x_max + 0.05*float(ground.X[-1]), 100.0)
y_upper = max(2.30*H, y_max + 0.05*H, 100.0)
ax.set_xlim(min(0.0 - 0.05*float(ground.X[-1]), -2.0), x_upper)
ax.set_ylim(0.0, y_upper)
ax.set_aspect("equal", adjustable="box")
ax.grid(True); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

from matplotlib.patches import Patch
legend_patches=[Patch(color=(0.85,0,0),label="Fs<1.0"),
                Patch(color=(1.0,0.75,0.0),label="1.0≤Fs<1.2"),
                Patch(color=(0.0,0.55,0.0),label="Fs≥1.2")]
h,l = ax.get_legend_handles_labels()
ax.legend(h+legend_patches, l+[p.get_label() for p in legend_patches], loc="upper right", fontsize=9)

tail=[f"MinFs={refined[idx_minFs]['Fs']:.3f}", f"TargetFs={Fs_target:.2f}", f"Water={water_mode}"]
if water.get("type")=="ru": tail.append(f"ru={water.get('ru'):.2f}")
st.caption(" / ".join(tail))

st.pyplot(fig, use_container_width=True); plt.close(fig)

# metrics
m1,m2 = st.columns(2)
with m1: st.metric("Min Fs（精密・選抜センター）", f"{refined[idx_minFs]['Fs']:.3f}")
with m2: st.metric("Max required T", f"{refined[idx_maxT]['T_req']:.1f} kN/m")
