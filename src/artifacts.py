"""JSON 产物的数值清洗与原子写入，避免读到半份 JSON。"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path


def finite_float(value, digits: int | None = None) -> float | None:
    """JSON 不支持 NaN/Infinity；缺失或不可用的指标返回 null。"""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits) if digits is not None else number


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return finite_float(value)
    return value


def write_json_atomic(path: Path, value: dict) -> None:
    """同目录临时文件写完后替换，写入失败时保留上一版产物。"""
    payload = json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(payload)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
