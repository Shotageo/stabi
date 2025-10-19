# >>> DXF_PLAN_PREVIEW START（追記：DXFプレビュー） >>>
with st.expander("🗺️ DXF：中心線＋横断群のプレビュー（実験）", expanded=False):
    st.caption("DXFから Alignment（中心線形）と XS*（横断法線）を読み込み、平面図に重ねて表示します。")
    dxf_file = st.file_uploader("DXFファイルを選択", type=["dxf"], key="__dxf_plan__")
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        layer_align = st.text_input("中心線レイヤ名ヒント", value="ALIGN")
    with colB:
        layer_xs = st.text_input("横断レイヤ名（接頭辞OK）", value="XS")
    with colC:
        highlight = st.text_input("強調表示する横断ID（任意）", value="")

    try:
        if dxf_file is not None:
            import tempfile, os
            from io.dxf_sections import load_alignment, load_sections, attach_stationing
            from viz.plan_preview import plot_plan_preview
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tf:
                tf.write(dxf_file.read())
                dxf_path = tf.name
            try:
                ali = load_alignment(dxf_path, layer_hint=layer_align.strip() or None)
                xs_raw = load_sections(dxf_path, layer_filter=layer_xs.strip() or "XS")
                xs = attach_stationing(xs_raw, ali)
                if not xs:
                    st.warning("横断レイヤ（XS*）が見つかりません。")
                else:
                    st.success(f"読み込み成功：Alignment={ali.length:.1f} m、横断本数={len(xs)}")
                    fig2, ax2 = plt.subplots(figsize=(8.6, 6.0))
                    plot_plan_preview(ax2, ali, xs, highlight_id=(highlight or None))
                    st.pyplot(fig2)
                    plt.close(fig2)
                    st.caption("※ プレビューのみ。解析やcfgには影響しません。")
            finally:
                try:
                    os.unlink(dxf_path)
                except Exception:
                    pass
        else:
            st.info("DXFを選択すると平面図プレビューが表示されます。")
    except Exception as e:
        st.error(f"DXFプレビューでエラーが発生しました：{e}")
# <<< DXF_PLAN_PREVIEW END <<<
