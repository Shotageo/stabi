# stabi_viz/__init__.py  — module型の再エクスポート（module.page で呼べる）

# 旧UIモジュールもそのまま出す（あるなら）
from . import plan_preview as plan_preview

# 新UIは「モジュール」として再エクスポートする
from . import plan_preview_upload as plan_preview_upload

__all__ = ["plan_preview", "plan_preview_upload"]

