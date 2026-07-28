"""FastAPI 后端包。

把项目根目录加入 sys.path，以便复用 config 与 src 里的工具函数
（只引用、不改动 src 内任何代码）。
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
