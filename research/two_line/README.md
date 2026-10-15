# 两条ETF研究路线的独立周频研究

本目录只生成独立研究产物，不发布生产模型，不改旧冻结协议和历史结论。中文计划、当前收敛和风险缺口见[复核报告](../../docs/TWO_LINE_REVIEW_20261013.md)。

## 研究入口

第一线保持原模型与组合规则，改变决策日程为每周首交易日。须先完成独立扩展训练，再指定完成目录：

```powershell
.venv/Scripts/python.exe -m research.prod_candidate.run
.venv/Scripts/python.exe -m research.two_line.refresh --source output/prod_candidate/runs/已完成运行目录
```

第二线预登记一个跨行业动量60/趋势200/波动15候选，与固定通信同现金账本比较；先登记再运行，全部失败产物保留：

```powershell
.venv/Scripts/python.exe -m research.two_line.dynamic_industry check
.venv/Scripts/python.exe -m research.two_line.dynamic_industry register
.venv/Scripts/python.exe -m research.two_line.dynamic_industry run --directory 已登记目录
.venv/Scripts/python.exe -m unittest research.two_line.test_dynamic_industry -v
```

`refresh_protocol.json`和`dynamic_protocol.json`分别保存两个独立实验。旧控制与新现金引擎并不完全相同，报告时必须区分执行版本。真正十年LightGBM仍缺更早训练与当时公告规模；不补造2016/2017预测。

本次结论：LightGBM是三种原排序模型的相对优选，固定结构按风险偏好保留trend200_vol15研究锚点；本次跨行业版本风险收益均较差，予以否决。两路线当前均未获实盘准入，尚无满足可切换结构与三项联合目标的第二线赢家。
