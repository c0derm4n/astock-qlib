"""沪深300 PE / 创业板 PB 估值择时独立研究实现。

边界: 只写 ``research/valuation_bands``、``data/valuation_bands``、``output/valuation_bands``;
不修改生产策略、全局ETF池、正式数据库或旧冻结实验。
"""
from __future__ import annotations

__all__ = ["data", "calibration", "signals", "engine", "verify", "synthetic"]
__version__ = "1.0.0"
