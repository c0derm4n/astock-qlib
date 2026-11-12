"""把已复算的八项结果嵌入对话图表，不从图形样本重算绩效。"""
from html import escape
import json
from pathlib import Path

from .compare_100k import OUT, ROOT, todo_write
from .compare_dividend_index import write_json

VIS = Path('C:/Users/Administrator/.codex/visualizations/2026/11/09/01a1bec6-b69d-7ef3-8e4a-b27766cfbdaf')


def main():
    VIS.mkdir(parents=True, exist_ok=True)
    data = json.loads((OUT / 'visualization_data.json').read_text(encoding='utf-8'))
    columns = ['date'] + [data['labels'][k] for k in data['series_ids']]
    configs = {}
    for mode in ['wealth', 'drawdown']:
        values = data['wealth_wan'] if mode == 'wealth' else data['drawdowns']
        rows = [[d] + [values[k][i] for k in data['series_ids']] for i, d in enumerate(data['dates'])]
        config = {'id': 'strategy-wealth-series', 'title': '10万元起始账户金额' if mode == 'wealth' else '历史回撤',
                  'showHeader': False, 'height': 370,
                  'description': '八项历史曲线，2016-09-12至2026-11-04。周末观察用于显示，风险指标来自完整日频数据。',
                  'columns': columns, 'rows': rows,
                  'spec': {'type': 'line', 'x': 'date', 'y': columns[1:],
                           'xLabel': '日期', 'yLabel': '金额（万元）' if mode == 'wealth' else '回撤（%）',
                           'showLegend': True, 'startAtZero': True,
                           'valueFormat': {'style': 'decimal' if mode == 'wealth' else 'percent',
                                           'minimumFractionDigits': 2, 'maximumFractionDigits': 2}}}
        configs[mode] = config
        write_json(OUT / f'chart_{mode}_config.json', config)
    table = []
    for row in data['rows']:
        table.append('<tr><th scope="row">' + escape(row['label']) + '</th>' +
                     f'<td class="text-end">{row["final_nav"]/10000:.2f}</td>' +
                     f'<td class="text-end">{row["cagr"]:.2%}</td>' +
                     f'<td class="text-end">{row["worst_full_year"]:.2%}<br>（{row["worst_year"]}）</td>' +
                     f'<td class="text-end">{row["max_drawdown"]:.2%}</td></tr>')
    fragment = (ROOT / 'research/valuation_bands/comparison_inline_template.html').read_text(encoding='utf-8')
    fragment = fragment.replace('__TABLE_ROWS__', '\n'.join(table))
    for token, config in [('__WEALTH_CONFIG__', configs['wealth']), ('__DRAWDOWN_CONFIG__', configs['drawdown'])]:
        fragment = fragment.replace(token, json.dumps(config, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c'))
    target = VIS / 'strategy-wealth-100k.html'
    target.write_text(fragment, encoding='utf-8')
    assert target.stat().st_size < 1_000_000
    assert '512890' not in fragment and '__TABLE_ROWS__' not in fragment
    assert len(configs['wealth']['rows']) == len(configs['drawdown']['rows']) == 520
    for row in configs['wealth']['rows'][:1]:
        assert all(abs(x-10) < 1e-10 for x in row[1:])
    for j, r in enumerate(data['rows']):
        assert abs(configs['wealth']['rows'][-1][j+1]*10000-r['final_nav']) < 1e-7
    receipt = {'schemaVersion': 1, 'items': [{
        'id': 'strategy-comparison', 'title': '10万元起始的八项历史比较',
        'queries': [{
            'id': 'historical-curves',
            'source': {'label': '项目归档净值、中证指数、Yahoo Finance',
                       'files': [{'label': 'comparison.json'}, {'label': 'daily_wealth_native_calendars.csv'},
                                 {'label': 'annual_returns.csv'}, {'label': 'verification.json'}],
                       'links': [{'label': '中证沪深300指数资料', 'href': 'https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000300factsheet.pdf'},
                                 {'label': '中证红利低波指数资料', 'href': 'https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/H30269factsheet.pdf'},
                                 {'label': '标普全收益定义', 'href': 'https://www.spglobal.com/spdji/en/education/article/faq-sp-500-dividend-points-index/'},
                                 {'label': '标普500全收益日线', 'href': 'https://finance.yahoo.com/quote/%5ESP500TR/history/'},
                                 {'label': 'USD/CNY历史汇率', 'href': 'https://finance.yahoo.com/quote/CNY=X/history/'}],
                       'metricDefinitions': [
                           {'label': '起始资金', 'definition': '所有曲线于2016年9月12日归一化为10万元，期末金额以人民币万元显示。'},
                           {'label': '最差完整年', 'definition': '最差完整年取2017—2025完整自然年收益的最低值，2016和2026不参与。'},
                           {'label': '最大回撤', 'definition': '最大回撤按各系列全部日频净值相对此前历史峰值的最大损失计算。'}],
                       'filters': ['排除512890上市前现金对照', '四条策略、510880再投代理与三条全收益指数'],
                       'caveats': ['策略曲线属于历史回测，并非从2016年开始的实盘账户业绩。',
                                   '10万元为原净值的等比例展示，没有按10万元整手成交与最低佣金重新回放。',
                                   '标普500按USD/CNY换算为人民币，全收益含股息再投，未扣税和汇兑成本。',
                                   '中美曲线按当地收盘日期作事后比较，不代表同一北京时间；不参与策略信号。',
                                   '图中周末观察仅用于显示，表中风险指标来自全部日频净值；账户仓位和费用差异保留。',
                                   'PE/PB沿用用户授权的等权估值版本，历史重建非PIT，PB财务MRQ未经证实。'],
                       'evidenceFlow': [
                           {'kind': 'validation', 'title': '交易日覆盖', 'detail': '六条原序列及沪深300覆盖2460个中国交易日；标普覆盖2552个美股交易日，人民币估值保留2644个美股或汇率报价日。'},
                           {'kind': 'validation', 'title': '数值复核', 'detail': '六条原序列的年化、最差完整年与回撤和归档值逐一一致；八条净值均以独立收益连乘、年度年末比值和逐日峰值回撤复核。'}]},
            'reportingPeriod': '2016-09-12—2026-11-04；完整自然年2017—2025',
            'columns': [{'field': 'label', 'label': '策略或基准'}, {'field': 'final_nav', 'label': '期末人民币金额（元）'},
                        {'field': 'cagr', 'label': '年化收益（比例）'}, {'field': 'worst_full_year', 'label': '最差完整年收益（比例）'},
                        {'field': 'worst_year', 'label': '最差年份'}, {'field': 'max_drawdown', 'label': '最大回撤（比例）'}],
            'rows': [{k: r[k] for k in ['label', 'final_nav', 'cagr', 'worst_full_year', 'worst_year', 'max_drawdown']} for r in data['rows']],
            'methods': [{'language': 'calculation', 'code': '展示金额 = 历史净值 / 起始日净值 × 100000\n标普人民币指数值 = 标普全收益美元指数值 × 当日USD/CNY\n年化 = (期末金额 / 100000) ** (365.2425 / 区间天数) - 1\n最大回撤 = max(1 - 每日净值 / 截至当日最高净值)'}]
        }]}]}
    write_json(OUT / 'reviewed_sources.json', receipt)
    todo_write('verification')
    print(json.dumps({'visualization': str(target), 'bytes': target.stat().st_size,
                      'chart_payloads': 2, 'series': len(data['rows'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
