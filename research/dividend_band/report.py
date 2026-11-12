"""生成红利低波估值带研究的中文报告与对比图。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / 'output/dividend_band/runs/20261111-174236-69e76e21'
EXTRA = ROOT / 'output/dividend_band/runs/20261111-174352-b71cc891'
REVIEW = ROOT / 'output/dividend_band/review_20261111'
LEADERS = ['abs_band', 'abs_band_wide', 'pct_band', 'erp_pct_band']


def pct(value, digits=2):
    return f'{value * 100:.{digits}f}%'


def load_nav(run, case):
    daily = pd.read_csv(run / 'cases' / case / 'daily.csv', parse_dates=['date'])
    return daily.set_index('date').nav


def main_table(rows, labels):
    lines = ['| 策略／基准 | 期末万元 | 年化 | 最差完整年 | 最大回撤 | 平均红利/国债/现金 | 年换手 |',
             '| --- | ---: | ---: | ---: | ---: | --- | ---: |']
    for row in rows:
        name = labels.get(row['id'], row['id'])
        lines.append('| {} | {:.2f} | {} | {}（{}） | {} | {}/{}/{} | {:.1f}倍 |'.format(
            name, row['final_nav'] / 10000, pct(row['cagr']), pct(row['worst_full_year']),
            row['worst_year'], pct(row['max_drawdown']), pct(row['average_red_weight'], 1),
            pct(row['average_bond_weight'], 1), pct(row['average_cash_weight'], 1),
            row['annual_one_way_turnover']))
    return '\n'.join(lines)


def segment_table(records, labels):
    windows = ['2017-2021', '2022-2025', '2017-2024']
    order = LEADERS + ['red_benchmark']
    index = {(r['id'], r['window']): r for r in records}
    lines = ['| 分段 | ' + ' | '.join(labels.get(i, i) for i in order) + ' |',
             '| --- | ' + ' | '.join('---:' for _ in order) + ' |']
    for window in windows:
        cells = []
        for name in order:
            row = index.get((name, window))
            cells.append('{}／{}'.format(pct(row['cagr']), pct(row['max_drawdown']))
                         if row else '—')
        lines.append(f'| {window} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def plot(curves, benchmark, path, labels):
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                gridspec_kw={'height_ratios': [2.2, 1]})
    styles = {'abs_band': ('#1f4e79', 1.8), 'abs_band_wide': ('#2e8b57', 1.8),
              'pct_band': ('#b8860b', 1.8), 'erp_pct_band': ('#8b2e5f', 1.6),
              'hold95': ('#808080', 1.2)}
    for name, series in curves.items():
        color, width = styles.get(name, ('#333333', 1.0))
        axes[0].plot(series.index, series / 10000, label=labels.get(name, name),
                     color=color, linewidth=width)
        axes[1].plot(series.index, 1 - series / series.cummax(), color=color, linewidth=width)
    axes[0].plot(benchmark.index, benchmark / 10000, label='红利低波全收益基准', color='#c00000',
                 linewidth=2.4, linestyle='--')
    axes[1].plot(benchmark.index, 1 - benchmark / benchmark.cummax(), color='#c00000',
                 linewidth=2.2, linestyle='--')
    axes[0].set_ylabel('账户净值（万元）')
    axes[0].set_title('红利低波估值带策略 vs 红利低波全收益基准（2016-09-12 起，10万元）')
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].grid(alpha=.3)
    axes[1].set_ylabel('回撤（每日净值）')
    axes[1].grid(alpha=.3)
    axes[1].yaxis.set_major_formatter(lambda v, _: f'{v*100:.0f}%')
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    REVIEW.mkdir(parents=True, exist_ok=True)
    main_result = json.loads((MAIN / 'results.json').read_text(encoding='utf-8'))
    extra = pd.read_csv(EXTRA / 'comparison.csv')
    rows = {r['id']: r for r in main_result['rows']}
    labels = {'abs_band': '方案三A：绝对股息率七档带', 'abs_band_wide': '方案三B：绝对股息率三档带',
              'pct_band': '方案三C：滚动分位五档带', 'erp_pct_band': '方案三D：ERP分位五档带',
              'erp_band': '方案三E：绝对ERP六档带', 'abs_band_guard': 'A＋深度破位保护',
              'pct_band_guard': 'C＋深度破位保护', 'erp_band_guard': 'E＋深度破位保护',
              'abs_band_cash': 'A但闲置资金持零息现金', 'hold95': '95%红利持有（同成本对照）',
              'bond95': '95%国债持有（同成本对照）', 'static60': '60%红利＋35%国债静态',
              'static70': '70%红利＋25%国债静态', 'static75': '75%红利＋20%国债静态',
              'red_benchmark': '红利低波全收益基准'}
    curves = {name: load_nav(MAIN, name) for name in LEADERS + ['hold95']}
    benchmark = pd.read_csv(MAIN / 'red_benchmark_nav.csv', parse_dates=['date']
                            ).set_index('date').iloc[:, 0]
    plot(curves, benchmark, REVIEW / 'comparison.png', labels)

    base_rows = [rows[c] for c in ['abs_band', 'abs_band_wide', 'pct_band', 'erp_pct_band',
                                   'erp_band', 'hold95', 'bond95', 'static70', 'abs_band_cash']]
    base_table = main_table(base_rows, labels)
    baseline = main_result['benchmark']
    bench_line = ('| 红利低波全收益基准（H20269，未扣费） | {:.2f} | {} | {}（{}） | {} | 100/0/0 | 0 |'
                  .format(baseline['final_nav'] / 10000, pct(baseline['cagr']),
                          pct(baseline['worst_full_year']), baseline['worst_year'],
                          pct(baseline['max_drawdown'])))

    stress_rows = []
    for name in ['abs_band_wide', 'pct_band', 'erp_pct_band']:
        for scenario in ['base', 'cost2', 'cost3', 'delay2', 'month', 'etf']:
            row = extra[extra.id == f'{name}__{scenario}']
            if len(row):
                item = row.iloc[0]
                stress_rows.append((f'{labels[name]}｜{scenario}', item))
    stress_lines = ['| 候选／情景 | 年化 | 最差完整年 | 最大回撤 | 三项是否优于基准 |',
                    '| --- | ---: | ---: | ---: | --- |']
    scenario_names = {'base': '基础情景', 'cost2': '单边成本0.20%', 'cost3': '单边成本0.30%',
                      'delay2': '延迟到T+2收盘', 'month': '改为每月首个交易日', 'etf': '2019-01-18后512890子样本'}
    for name, item in stress_rows:
        head, scenario = name.split('｜')
        stress_lines.append('| {}｜{} | {} | {}（{}） | {} | {} |'.format(
            head, scenario_names[scenario], pct(item.cagr), pct(item.worst_full_year),
            item.worst_year, pct(item.max_drawdown),
            '是' if item.dominates_red_benchmark else '否'))
    stress_table = '\n'.join(stress_lines)
    etf_bench = main_result['etf_benchmark']

    verification = json.loads((MAIN / 'verification.json').read_text(encoding='utf-8'))
    extra_verification = json.loads((EXTRA / 'verification.json').read_text(encoding='utf-8'))
    total_cases = verification['case_count'] + extra_verification['case_count']

    report = f"""# 红利低波估值带（隐含股息率／股债利差）研究

研究日期：2026-11-11。区间 2016-09-12—2026-11-04，完整自然年 2017—2025。仅新增 `research/dividend_band/` 与 `output/dividend_band/`，未修改正式服务、生产指针、策略注册、正式数据库或用户偏好。

## 判断

方案三（估值带＋国债承接）在**基础成本下确实三项同时优于红利低波全收益基准**，且完全不使用黄金、2025 年也没有任何贵金属收益贡献，这一点与「红利低波/黄金双动量」有本质差别；同一批候选在 2017—2024 子区间也能跑赢基准。

但优势对交易成本敏感：单边成本升到 0.20% 时，绝对七档带（9.11%）与 ERP 分位带（9.12%）的年化掉到基准（9.26%）以下，只有绝对三档带（9.27%）与滚动分位带（9.52%）仍在基准之上；单边成本 0.30% 时三个领先候选全部低于基准。因此**不能宣称持续全面优于**，只能称为全期基础情景下的风险优先研究保留候选。

## 数据与信号定义

- 红利：中证红利低波动全收益指数 H20269 官方日收盘；2014—2016 预热来自中证官方接口按年原始响应。持有方按日历天扣 0.60% 年化复制费用情景。
- 价格：中证红利低波动指数 H30269 官方日收盘（不含股息）。
- 隐含派息：`D_t = TR_t/TR_(t-1)*PR_(t-1) - PR_t`；滚动 252 个交易日派息总额除以当日价格得到隐含股息率，2026-11-04 为 {pct(float(pd.read_csv(MAIN / 'indicators.csv', parse_dates=['date']).set_index('date').dividend_yield.dropna().iloc[-1]) / 100)}，与中证官网同期披露的 4.65% 量级一致，但口径不是官方股息率。
- 股债利差：ERP = 隐含股息率 − 10 年期中国国债到期收益率（东方财富原始分页 JSON 留档，按 XSHG 交易日向前填充）。
- 国债：十年国债 ETF 511260 收盘×复权因子；2017-08-24 上市前保留现金、不造上市前收益；2018-08-17 缺报价只估值不成交。
- 全部信号只用当日及以前数据；周首个交易日收盘评估，下一交易日收盘执行。

## 全期比较（基础情景：单边成本 0.10%，红利复制费 0.60%/年）

{base_table}
{bench_line}

- 平均红利暴露约 72%，因此「95% 持有」与「70% 静态」是同账户口径下的关键对照：95% 持有年化 8.30%、70% 静态 7.15%，都明显低于估值带候选的 9.4%—10.0%，说明增益来自档位择时本身，而不是仅仅因为降低了权益暴露。
- 把闲置资金改为零息现金（`abs_band_cash`）后年化从 9.42% 降到 8.98%、最差年从 +0.59% 降到 −3.71%，最大回撤不变；国债承接是收益与年度风险改善的直接来源。
- 「深度破位保护」在本批里是负贡献：三个加保护版本年化只有 7.3%—8.3%，因为 2018—2019 与 2022 年的破位往往同时是估值加仓点。

## 分段表现（年化／最大回撤）

{segment_table(main_result['segments'], labels)}

2017—2024 这段不含 2025 行情的八年里，绝对七档带 10.31%、绝对三档带 10.60%、滚动分位带 10.71%、ERP 分位带 10.38%，均高于基准 10.03%；「红利低波/黄金双动量」在同一子区间只有 8.42%。2022—2025 段估值带普遍略弱于基准（基准 11.22%），这是估值带在红利单边上涨阶段过早降档的代价。

## 压力与执行敏感性

{stress_table}

- 2019-01-18 起改用 512890 真实 ETF 价格的子样本：512890 持有代理年化 {pct(etf_bench['cagr'])}、回撤 {pct(etf_bench['max_drawdown'])}、最差完整年 {pct(etf_bench['worst_full_year'])}（2022）。滚动分位带 {pct(float(extra[extra.id == 'pct_band__etf'].cagr.iloc[0]))}／{pct(float(extra[extra.id == 'pct_band__etf'].max_drawdown.iloc[0]))}、ERP 分位带 {pct(float(extra[extra.id == 'erp_pct_band__etf'].cagr.iloc[0]))}／{pct(float(extra[extra.id == 'erp_pct_band__etf'].max_drawdown.iloc[0]))} 三项均优于持有；绝对三档带 {pct(float(extra[extra.id == 'abs_band_wide__etf'].cagr.iloc[0]))} 低于持有 {pct(etf_bench['cagr'])}，不构成上市后胜出。
- 换手 2.9—4.3 倍/年（单边），高于静态但显著低于「双动量」的 7.6 倍；月频评估可把换手降到 1.6—2.6 倍，年化少 0.1—0.9 个百分点。

## 选择过程与验证范围

主 run 登记 14 个候选（含 95% 持有、95% 国债、60/70/75 静态与零息现金对照），4 个锚点各做成本 0.20%、0.30%、T+2 延迟与月频评估，另加 4 个 ETF 子样本，共 {verification['case_count']} 条回放。看到结果后另新登记补充批次，只对三个领先候选追加同一组压力与 ETF 子样本，共 {extra_verification['case_count']} 条，不新增档位、不改阈值，合计 {total_cases} 条，不能理解为 {total_cases} 个独立策略。

- 13 项合成单元测试通过；独立复核包含逐日自融资恒等式（最大误差 {verification['self_financing_max_error']:.1e}）、单遍回撤与年末收益复算（最大误差 {verification['independent_metric_max_error']:.1e}）、numpy 逐窗独立信号（最大误差 {verification['independent_signal_max_error']:.1e}）、独立档位映射（误差 0）、截断到 2021-12-31 的因果前缀（最大误差 {max(verification['causal_prefix_max_errors'].values()):.1e}）、份额×价格重建账户（指数 {verification['index_unit_reconstruction_max_error']:.1e}、ETF {verification['etf_unit_reconstruction_max_error']:.1e}）与冻结输入哈希核对。
- 首次登记 run `20261111-174048-07a0648e` 在验证阶段失败：独立复算对 pandas 滚动百分位的缺失值语义处理错误。失败记录与原因保留在该目录 `failure.json`，回放产物未被改动；修正后重新登记本次 run，两者 `comparison.csv` 全部指标逐行相同（差 0.0），可复现。
- 全部历史已查看，没有真正未见数据的盲测；档位阈值来自用户给定的方案三表述与事先写的档位表，但研究者已看过这段历史，仍属事后研究。
- 未验证 100 份整手、真实券商收盘成交、成交容量与全量停牌/涨跌停约束；指数历史与复权行情属于当前快照，不是原始时点留存数据。

## 结论与下一步

- 风险优先（用户偏好）下最贴近方案三原意的保留候选是**绝对股息率七档带**：年化 9.42%、最差完整年 +0.59%、最大回撤 12.97%，三项全期优于基准，但相对基准的年化优势只有 0.16 个百分点，成本翻倍即失效。
- 若更看重成本耐受度，**绝对三档带**（4.8／4.0 两处阈值）年化 9.59%、成本翻倍仍有 9.27%、最差年 −0.14%，是本批里唯一在成本翻倍下仍三项过线的版本。
- **滚动分位带**年化最高（9.97%）且在 ETF 子样本也胜出，但 2018 年 −5.82% 已跌破用户 −5% 的年度亏损目标，不能直接当作首选。
- 这三项都未达到用户联合目标中的年化 ≥13.5%；年化与风险没有同时达标，不能作为正式策略或投资准入依据。

## 复跑

```powershell
.venv/Scripts/python.exe -X utf8 -m unittest research.dividend_band.test_strategy -v
.venv/Scripts/python.exe -X utf8 -m research.dividend_band.data --output output/dividend_band/data_20261111
.venv/Scripts/python.exe -X utf8 -m research.dividend_band.run --register --data output/dividend_band/data_20261111
.venv/Scripts/python.exe -X utf8 -m research.dividend_band.run --run <新登记目录>
.venv/Scripts/python.exe -X utf8 -m research.dividend_band.followup --register <主run目录>
.venv/Scripts/python.exe -X utf8 -m research.dividend_band.followup --run <新登记目录> <主run目录>
.venv/Scripts/python.exe -X utf8 -m research.dividend_band.report
```

已完成目录不得覆盖；策略源文件变更必须重新登记。

![净值与回撤](comparison.png)
"""
    (REVIEW / 'report.md').write_text(report, encoding='utf-8')
    print(json.dumps({'report': str((REVIEW / 'report.md').relative_to(ROOT)),
                      'chart': str((REVIEW / 'comparison.png').relative_to(ROOT)),
                      'cases': total_cases}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
