"""第 3 步：训练 ETF 轮动模型并回测。

正式训练入口：固定研究协议、完整压力验收，并产出独立生产模型。
具体流程见 src.research。
"""
from __future__ import annotations


def main() -> None:
    """正式训练入口：固定研究协议、完整压力验收和独立生产模型。"""
    from src.research import main as research_main
    research_main()


if __name__ == "__main__":
    main()
