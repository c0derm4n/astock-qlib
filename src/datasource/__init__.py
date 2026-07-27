"""数据访问层：Tushare 主源 + AKShare 校验/补数，统一落 DuckDB 规范化真源。

仅在此暴露轻量的 DataStore（只依赖 duckdb）；数据源模块（依赖 tushare/akshare）
按需从各自模块显式导入，避免训练/回测环节被动加载重型依赖。
"""
from __future__ import annotations

from src.datasource.store import DataStore

__all__ = ["DataStore"]
