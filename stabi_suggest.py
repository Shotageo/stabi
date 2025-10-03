# stabi_suggest.py — Lint & Suggestion engine（Stabi教授コメント）
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Callable, Optional
import streamlit as st
import numpy as np

# ---------------- Types ----------------
@dataclass
class Action:
    id: str      # "set_mu:0.3" のように 'cmd:arg' 形式を推奨
    label: str   # ボタン表示

@dataclass
class Suggestion:
    level: str           # "info" | "warn" | "block"
    text: str            # 教授コメント
    actions: List[Action]
    tags: Tuple[str, ...] = tuple()

# ---------------- Render ----------------
def _box(level: str):
    if level == "block": return st.error
    if level == "warn":  return st.warning
    return st.info

def render_suggestions(suggs: List[Suggestion], key_prefix: str, dispatcher: Callable[[str], None]):
    for i, s in enumerate(suggs):
        box = _box(s.level)
        with box(s.text):
            cols = st.columns(max(1, len(s.actions)))
            for j, a in enumerate(s.actions):
                if cols[j].button(a.label, key=f"{key_prefix}:{i}:{j}:{a.id}"):
                    dispatcher(a.id)

# ---------------- Heuristics ----------------
def lint_geometry_and_water(ground, water_mode: str, ru: float, wl_points: Optional[np.ndarray]) -> List[Suggestion]:
    out: List[Suggestion] = []
    dx_total = float(ground.X[-1] - ground.X[0])
    if dx_total > 1e4:
        out.append(Suggestion(
            level="warn",
            text=f"地形の x 範囲が {dx_total:.0f} と大きいです。単位が mm かも。m へ換算して再計算しよう。",
            actions=[Action("scale_xy:0.001", "x,y を ×0.001 で換算")]
        ))
    if water_mode == "WT+ru":
        out.append(Suggestion(
            level="warn",
            text="WT と ru が同時ONです。どちらを優先しますか？同時は非推奨。",
            actions=[Action("water:WT", "WTを優先"), Action("water:ru", "ruを優先")]
        ))
    if water_mode.startswith("WT") and wl_points is not None:
        xs = wl_points[:,0]; ys = wl_points[:,1]
        yg = np.array([float(ground.y_at(x)) for x in xs])
        if np.any(ys > yg + 1e-6):
            out.append(Suggestion(
                level="warn",
                text="水位線が地表より上にあります。想定外なら地表にクリップしよう。",
                actions=[Action("wl:clip_to_ground", "水位を地表まで切下げる")]
            ))
    return out

def lint_soils_and_materials(soils: List[Dict[str, Any]], tau_grout_cap_kPa: Optional[float]) -> List[Suggestion]:
    out: List[Suggestion] = []
    if tau_grout_cap_kPa is None or tau_grout_cap_kPa <= 0:
        out.append(Suggestion(
            level="block",
            text="グラウトの付着上限 τ_grout_cap が未設定です。安全側に 150 kPa を仮置きしよう。",
            actions=[Action("set_tau_grout:150", "τ_grout_cap=150 kPa をセット")]
        ))
    for i, s in enumerate(soils, start=1):
        phi = float(s["phi"]); c = float(s["c"]); tau = float(s.get("tau_kPa", 0))
        if phi == 0 and c == 0:
            out.append(Suggestion(
                level="warn",
                text=f"Layer{i} の φ=0, c=0 は不安定になりがち。試算値で感度を見よう。",
                actions=[Action(f"soil{i}:set_cphi:5,25", f"Layer{i} を c=5 kPa, φ=25° に仮設定")]
            ))
        if tau > 300 and phi < 25:
            out.append(Suggestion(
                level="warn",
                text=f"Layer{i} の τ={tau:.0f} kPa はやや高め。粘性土でも 100–300 kPa が一般的。",
                actions=[Action(f"soil{i}:tau:200", f"Layer{i} τ=200 kPa で再評価")]
            ))
    return out

def lint_arc_and_slices(hit_edge: bool, n_slices: int) -> List[Suggestion]:
    out: List[Suggestion] = []
    if hit_edge:
        out.append(Suggestion(
            level="warn",
            text="臨界円弧が探索枠の端で止まっています。外側に20%拡張して再探索しよう。",
            actions=[Action("grid:expand20", "枠を20%拡張して再探索")]
        ))
    if n_slices < 30:
        out.append(Suggestion(
            level="info",
            text=f"スライス数 n={n_slices} はやや粗いです。FSの分散を抑えるため 40 を推奨。",
            actions=[Action("slices:40", "n_slices=40 で再計算")]
        ))
    return out

def lint_nails_layout(layout: Dict[str, Any], stats: Dict[str, Any]) -> List[Suggestion]:
    out: List[Suggestion] = []
    if stats.get("has_outward", False):
        out.append(Suggestion(
            level="block",
            text="打設角が外向きになっています。法線を地山側へ反転しましょう。",
            actions=[Action("nail:flip_outward", "向きを自動修正")]
        ))
    if stats.get("too_dense", False):
        out.append(Suggestion(
            level="warn",
            text="局所的にネイルが過密です（斜面実長ピッチが潰れています）。ピッチを広げよう。",
            actions=[Action("pitch:+0.5", "S_surf を +0.5 m")]
        ))
    n_short = int(stats.get("n_Lo_short", 0))
    if n_short > 0:
        out.append(Suggestion(
            level="warn",
            text=f"必要定着長 Lo_min 未満のネイルが {n_short} 本あります。全段 +0.5 m で補正しますか？",
            actions=[Action("length:+0.5", "全段 +0.5 m して再評価")]
        ))
    mode = stats.get("dominant_mode")
    if mode == "tensile":
        out.append(Suggestion(
            level="info",
            text="Tensile支配が多いです。鉄筋径を一段上げると余裕が出ます。",
            actions=[Action("bar:next", "鉄筋径を一段アップ")]
        ))
    if mode == "strip":
        out.append(Suggestion(
            level="info",
            text="Strip支配が多いです。μを下げるか内側長を +0.5 m しよう。",
            actions=[Action("mu:down", "μを 0.1 下げる"), Action("length:+0.5", "内側長 +0.5 m")]
        ))
    return out

# ---------------- Dispatcher ----------------
def default_dispatcher(action_id: str):
    cmd, *rest = action_id.split(":")
    arg = rest[0] if rest else ""

    if cmd == "scale_xy":
        st.session_state["_cmd_scale_xy"] = float(arg)

    elif cmd == "water":
        st.session_state["water_mode"] = arg  # "WT" or "ru"

    elif cmd == "wl" and arg == "clip_to_ground":
        st.session_state["_cmd_wl_clip"] = True

    elif cmd == "set_tau_grout":
        st.session_state["tau_grout_cap_kPa"] = float(arg)

    elif cmd.startswith("soil"):
        parts = action_id.split(":")
        head = parts[0]; subcmd = parts[1] if len(parts)>1 else None
        val = parts[2] if len(parts)>2 else None
        layer_idx = int(head.replace("soil",""))
        if subcmd == "set_cphi" and val:
            c, phi = map(float, val.split(","))
            st.session_state.setdefault("soils_override", {})[layer_idx] = {"c": c, "phi": phi}
        elif subcmd == "tau" and arg:
            st.session_state.setdefault("soils_override", {})[layer_idx] = {"tau_kPa": float(arg)}

    elif cmd == "grid" and arg == "expand20":
        st.session_state["_cmd_expand_grid"] = 0.20
    elif cmd == "slices":
        st.session_state["n_slices"] = int(arg)

    elif cmd == "nail" and arg == "flip_outward":
        st.session_state["_cmd_flip_nails"] = True
    elif cmd == "pitch" and arg == "+0.5":
        st.session_state["_cmd_pitch_delta"] = +0.5
    elif cmd == "length" and arg == "+0.5":
        st.session_state["_cmd_length_delta"] = +0.5
    elif cmd == "bar" and arg == "next":
        st.session_state["_cmd_bar_next"] = True
    elif cmd == "mu" and arg == "down":
        st.session_state["_cmd_mu_down"] = 0.1

    st.session_state["_recompute"] = True
