# =========================
# Page 3: 円弧探索（未補強） — フォーム化＋min/max依存排除
# =========================
elif page.startswith("3"):
    H, L, ground = HL_ground()
    st.subheader("円弧探索（未補強）")

    # 現在値（セッションから取得）
    x_min0 = float(st.session_state.get("x_min_abs", 0.25*L))
    x_max0 = float(st.session_state.get("x_max_abs", 1.15*L))
    y_min0 = float(st.session_state.get("y_min_abs", 1.60*H))
    y_max0 = float(st.session_state.get("y_max_abs", 2.20*H))
    pitch0 = float(st.session_state.get("grid_pitch_m", 5.0))
    method0 = st.session_state.get("method", "Bishop (simplified)")
    quality0 = st.session_state.get("quality", "Normal")
    Fs_t0 = float(st.session_state.get("Fs_target", 1.20))
    allow2_0 = bool(st.session_state.get("allow_cross2", True))
    allow3_0 = bool(st.session_state.get("allow_cross3", True))

    # ---- フォーム開始：ここで確定ボタンを押した時だけ state を更新 ----
    with st.form("arc_params"):
        colA, colB = st.columns([1.3, 1])
        with colA:
            # min/max は付けない（H/L揺れで強制クリップされないように）
            x_min = st.number_input("x min (m)", value=x_min0, step=max(0.1, 0.05*L), format="%.3f")
            x_max = st.number_input("x max (m)", value=x_max0, step=max(0.1, 0.05*L), format="%.3f")
            y_min = st.number_input("y min (m)", value=y_min0, step=max(0.1, 0.10*H), format="%.3f")
            y_max = st.number_input("y max (m)", value=y_max0, step=max(0.1, 0.10*H), format="%.3f")
            pitch = st.number_input("Center-grid ピッチ (m)", value=pitch0, min_value=0.1, step=0.1, format="%.2f")
            st.caption(f"ヒント: 参考レンジ x∈[{0.2*L:.1f},{4.0*L:.1f}], y∈[{0.8*H:.1f},{8.0*H:.1f}]")
        with colB:
            method = st.selectbox("Method", ["Bishop (simplified)","Fellenius"], index=["Bishop (simplified)","Fellenius"].index(method0))
            quality = st.select_slider("Quality", options=list(QUALITY.keys()), value=quality0)
            Fs_t = st.number_input("Target FS", value=Fs_t0, min_value=1.00, max_value=2.00, step=0.05, format="%.2f")

        allow2 = allow2_0
        allow3 = allow3_0
        if st.session_state.get("n_layers",3) >= 2:
            allow2 = st.checkbox("Allow into Layer 2", value=allow2_0)
        if st.session_state.get("n_layers",3) >= 3:
            allow3 = st.checkbox("Allow into Layer 3", value=allow3_0)

        submitted = st.form_submit_button("🔁 設定を確定（保存）")

    # 確定時だけ state に反映（method 切替などの rerun では上書きしない）
    if submitted:
        # 入力の安全化（範囲の整合）
        if x_max < x_min: x_min, x_max = x_max, x_min
        if y_max < y_min: y_min, y_max = y_max, y_min
        st.session_state["x_min_abs"] = float(x_min)
        st.session_state["x_max_abs"] = float(x_max)
        st.session_state["y_min_abs"] = float(y_min)
        st.session_state["y_max_abs"] = float(y_max)
        st.session_state["grid_pitch_m"] = float(max(0.1, pitch))
        st.session_state["method"] = method
        st.session_state["quality"] = quality
        st.session_state["Fs_target"] = float(Fs_t)
        st.session_state["allow_cross2"] = bool(allow2)
        st.session_state["allow_cross3"] = bool(allow3)
        st.success("設定を保存しました。")

    # 以降は、保存済みの値だけを使う（揺れない）
    ground = make_ground_example(st.session_state["H"], st.session_state["L"])
    H, L = st.session_state["H"], st.session_state["L"]
    x_min = float(st.session_state["x_min_abs"]); x_max = float(st.session_state["x_max_abs"])
    y_min = float(st.session_state["y_min_abs"]); y_max = float(st.session_state["y_max_abs"])
    pitch = float(st.session_state["grid_pitch_m"])
    method = st.session_state["method"]; quality = st.session_state["quality"]; Fs_t = float(st.session_state["Fs_target"])
    allow_cross=[]
    if st.session_state["n_layers"]>=2: allow_cross.append(bool(st.session_state["allow_cross2"]))
    if st.session_state["n_layers"]>=3: allow_cross.append(bool(st.session_state["allow_cross3"]))

    # プレビュー（地形・層・水位・センターグリッド）
    st.markdown("**設定プレビュー（横断図）**")
    interfaces = []
    if st.session_state["n_layers"] >= 2: interfaces.append(make_interface1_example(H, L))
    if st.session_state["n_layers"] >= 3: interfaces.append(make_interface2_example(H, L))

    Xd = np.linspace(ground.X[0], ground.X[-1], 600)
    Yg = np.array([float(ground.y_at(float(x))) for x in Xd], dtype=float)
    fig, ax = plt.subplots(figsize=(10.0, 6.8))
    if st.session_state["n_layers"]==1:
        ax.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Layer1")
    elif st.session_state["n_layers"]==2:
        Y1 = clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0]
        ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
        ax.fill_between(Xd, 0.0, Y1, alpha=0.12, label="Layer2")
    else:
        Y1,Y2 = clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)
        ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
        ax.fill_between(Xd, Y2, Y1, alpha=0.12, label="Layer2")
        ax.fill_between(Xd, 0.0, Y2, alpha=0.12, label="Layer3")
    ax.plot(ground.X, ground.Y, linewidth=2.0, label="Ground")
    if st.session_state["n_layers"]>=2: ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0], linestyle="--", linewidth=1.0)
    if st.session_state["n_layers"]>=3: ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)[1], linestyle="--", linewidth=1.0)

    wm = st.session_state.get("water_mode", "WT")
    wl = st.session_state.get("wl_points", None)
    if wm.startswith("WT"):
        if wl is None:
            # Page1を通っていなくても安全にWT生成
            Xw = np.linspace(ground.X[0], ground.X[-1], 200, dtype=float)
            Ygw = np.array([float(ground.y_at(x)) for x in Xw], dtype=float)
            Yw = np.clip(Ygw + st.session_state.get("wt_offset",-2.0), 0.0, Ygw)
            wl = np.vstack([Xw, Yw]).T
            st.session_state["wl_points"] = wl
        ax.plot(wl[:,0], wl[:,1], linestyle="-.", color="tab:blue", alpha=0.8, label="WT (clipped)")

    # center-grid（ピッチから生成）
    gx_mid = np.arange(x_min, x_max + 1e-9, pitch)
    gy_mid = np.arange(y_min, y_max + 1e-9, pitch)
    if gx_mid.size < 2: gx_mid = np.array([x_min, x_max])
    if gy_mid.size < 2: gy_mid = np.array([y_min, y_max])
    xs = [float(x) for x in gx_mid for _ in gy_mid]
    ys = [float(y) for y in gy_mid for _ in gx_mid]
    ax.scatter(xs, ys, s=10, c="k", alpha=0.25, marker=".", label=f"Center grid (pitch={pitch:.2f} m)")
    # 外枠
    ax.plot([x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min],
            color="k", linewidth=1.0, alpha=0.4)

    set_axes_fixed(ax, H, L, ground)
    ax.grid(True); ax.legend(loc="upper right"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    st.pyplot(fig); plt.close(fig)

    # Soils（stabi_lem.Soil は 3引数版）
    soils = [Soil(st.session_state["gamma1"], st.session_state["c1"], st.session_state["phi1"])]
    if st.session_state["n_layers"] >= 2:
        soils.append(Soil(st.session_state["gamma2"], st.session_state["c2"], st.session_state["phi2"]))
    if st.session_state["n_layers"] >= 3:
        soils.append(Soil(st.session_state["gamma3"], st.session_state["c3"], st.session_state["phi3"]))

    # 品質パラメータ
    P = QUALITY[quality].copy()

    def compute_once():
        # 毎回安全に ground/iface を構築（H/Lはstateから固定）
        Hc, Lc = st.session_state["H"], st.session_state["L"]
        ground_local = make_ground_example(Hc, Lc)
        interfaces_local = []
        if st.session_state["n_layers"] >= 2: interfaces_local.append(make_interface1_example(Hc, Lc))
        if st.session_state["n_layers"] >= 3: interfaces_local.append(make_interface2_example(Hc, Lc))

        def centers_by_pitch():
            xs = np.arange(x_min, x_max + 1e-9, pitch)
            ys = np.arange(y_min, y_max + 1e-9, pitch)
            if xs.size < 2: xs = np.array([x_min, x_max])
            if ys.size < 2: ys = np.array([y_min, y_max])
            return xs, ys

        def subsampled_centers():
            xs, ys = centers_by_pitch()
            tag = P["coarse_subsample"]
            if tag == "every 3rd":
                xs = xs[::3] if xs.size > 2 else xs
                ys = ys[::3] if ys.size > 2 else ys
            elif tag == "every 2nd":
                xs = xs[::2] if xs.size > 1 else xs
                ys = ys[::2] if ys.size > 1 else ys
            return [(float(xc), float(yc)) for yc in ys for xc in xs]

        def pick_center(budget_s):
            deadline = time.time() + budget_s
            best = None
            for (xc,yc) in subsampled_centers():
                cnt=0; Fs_min=None
                for _x1,_x2,_R,Fs in arcs_from_center_by_entries_multi(
                    ground_local, soils, xc, yc,
                    n_entries=P["coarse_entries"], method="Fellenius",
                    depth_min=0.5, depth_max=4.0,
                    interfaces=interfaces_local, allow_cross=allow_cross,
                    quick_mode=True, n_slices_quick=max(8, P["quick_slices"]//2),
                    limit_arcs_per_center=P["coarse_limit_arcs"],
                    probe_n_min=P["coarse_probe_min"],
                ):
                    cnt+=1
                    if (Fs_min is None) or (Fs < Fs_min): Fs_min = Fs
                    if time.time() > deadline: break
                score = (cnt, - (Fs_min if Fs_min is not None else 1e9))
                if (best is None) or (score > best[0]): best = (score, (xc,yc))
                if time.time() > deadline: break
            return (best[1] if best else None)

        center = pick_center(P["budget_coarse_s"])
        if center is None:
            return dict(error="Coarseで候補なし。枠/深さ/ピッチを見直してください。")
        xc, yc = center

        heap_R=[]; deadline=time.time()+P["budget_quick_s"]
        for _x1,_x2,R,Fs in arcs_from_center_by_entries_multi(
            ground_local, soils, xc, yc,
            n_entries=P["n_entries_final"], method="Fellenius",
            depth_min=0.5, depth_max=4.0,
            interfaces=interfaces_local, allow_cross=allow_cross,
            quick_mode=True, n_slices_quick=P["quick_slices"],
            limit_arcs_per_center=P["limit_arcs_quick"],
            probe_n_min=P["probe_n_min_quick"],
        ):
            heapq.heappush(heap_R, (-Fs, R))
            if len(heap_R) > max(P["show_k"], 20): heapq.heappop(heap_R)
            if time.time() > deadline: break
        R_candidates = [r for _fsneg, r in sorted([(-fsneg,R) for fsneg,R in heap_R], key=lambda t:t[0])]
        if not R_candidates:
            return dict(error="Quickで円弧候補なし。深さ/進入可/Quality/ピッチを調整してください。")

        refined=[]
        for R in R_candidates[:P["show_k"]]:
            Fs = fs_given_R_multi(ground_local, interfaces_local, soils, allow_cross, method, xc, yc, R, n_slices=P["final_slices"])
            if Fs is None: continue
            s = arc_sample_poly_best_pair(ground_local, xc, yc, R, n=251, y_floor=0.0)
            if s is None: continue
            x1,x2,*_ = s
            packD = driving_sum_for_R_multi(ground_local, interfaces_local, soils, allow_cross, xc, yc, R, n_slices=P["final_slices"])
            if packD is None: continue
            D_sum,_,_ = packD
            T_req = max(0.0, (Fs_t - Fs)*D_sum)
            refined.append(dict(Fs=float(Fs), R=float(R), x1=float(x1), x2=float(x2), T_req=float(T_req)))
        if not refined:
            return dict(error="Refineで有効弧なし。設定/Quality/ピッチを見直してください。")
        refined.sort(key=lambda d:d["Fs"])
        idx_minFs = int(np.argmin([d["Fs"] for d in refined]))
        return dict(center=(xc,yc), refined=refined, idx_minFs=idx_minFs)

    if st.button("▶ 計算開始（未補強）"):
        res = compute_once()
        if "error" in res: st.error(res["error"]); st.stop()
        st.session_state["res3"] = res
        # 採用円弧を保存（Page4/5用）
        xc,yc = res["center"]; d = res["refined"][res["idx_minFs"]]
        st.session_state["chosen_arc"] = dict(xc=xc,yc=yc,R=d["R"], x1=d["x1"], x2=d["x2"], Fs=d["Fs"])

    # 結果描画
    if st.session_state["res3"]:
        res = st.session_state["res3"]
        xc,yc = res["center"]; refined=res["refined"]; idx_minFs=res["idx_minFs"]

        fig, ax = plt.subplots(figsize=(10.0, 7.0))
        # 背景
        if st.session_state["n_layers"]==1:
            ax.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Layer1")
        elif st.session_state["n_layers"]==2:
            Y1 = clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0]
            ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1"); ax.fill_between(Xd, 0.0, Y1, alpha=0.12, label="Layer2")
        else:
            Y1,Y2 = clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)
            ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
            ax.fill_between(Xd, Y2, Y1, alpha=0.12, label="Layer2")
            ax.fill_between(Xd, 0.0, Y2, alpha=0.12, label="Layer3")
        ax.plot(ground.X, ground.Y, linewidth=2.0, label="Ground")

        # Refined arcs
        for d in refined[:30]:
            xs=np.linspace(d["x1"], d["x2"], 200)
            ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
            clipped = clip_yfloor(xs, ys, 0.0)
            if clipped is None: continue
            xs_c, ys_c = clipped
            ax.plot(xs_c, ys_c, linewidth=0.9, alpha=0.75, color=fs_to_color(d["Fs"]))

        # Min Fs
        d=refined[idx_minFs]
        xs=np.linspace(d["x1"], d["x2"], 400)
        ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
        clipped = clip_yfloor(xs, ys, 0.0)
        if clipped is not None:
            xs_c, ys_c = clipped
            ax.plot(xs_c, ys_c, linewidth=3.0, color=(0.9,0.0,0.0), label=f"Min Fs = {d['Fs']:.3f}")
            y1=float(ground.y_at(xs_c[0])); y2=float(ground.y_at(xs_c[-1]))
            ax.plot([xc,xs_c[0]],[yc,y1], linewidth=1.1, color=(0.9,0.0,0.0), alpha=0.9)
            ax.plot([xc,xs_c[-1]],[yc,y2], linewidth=1.1, color=(0.9,0.0,0.0), alpha=0.9)

        set_axes_fixed(ax, H, L, ground)
        ax.grid(True); ax.legend()
        ax.set_title(f"Center=({xc:.2f},{yc:.2f}) • MinFs={refined[idx_minFs]['Fs']:.3f} • TargetFs={Fs_t:.2f} • pitch={pitch:.2f}m")
        st.pyplot(fig); plt.close(fig)