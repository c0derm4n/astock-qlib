"""汇总原件、支付回放和收益归因验收；保留失败结论与原正式指针。"""
from pathlib import Path
import json
import sys

import numpy as np

from src.artifacts import write_json_atomic
from research.goal30.acquire_remaining_dividends import ROOT as EVIDENCE
from research.goal30.audit import BASE
from research.goal30.bond_dividend_evidence import digest
from research.goal30.group_return_attribution import OUT as ATTRIBUTION
from research.goal30.primary_research_series import read
from research.goal30.run_dividend_payment_replay import OUT as REPLAY
from research.goal30.turnover_budget import require

NAMES={'group4_risk_parity':'分组4风险均衡（主方案）','ungrouped4_equal':'无分组4等权',
       'group4_equal':'分组4等权','group3_risk_parity':'分组3风险均衡','group5_risk_parity':'分组5风险均衡',
       'fixed_control':'固定配置控制'}
SCENARIOS={'base':'基础','slippage_2':'滑点2倍','payment_delay_5':'支付延迟5日'}
PARTS={'cash_opportunity':'未投资资金机会成本','invested_excess':'持仓相对表现','trading_time':'交易时点',
       'commission':'佣金','slippage':'滑点与冲击'}


def create():
    directory=BASE/read(REPLAY/'latest_registration.json')['directory'];replay=read(directory/'results.json')
    proof=read(directory/'verification.json');evidence=read(EVIDENCE/'verified_remaining_dividends.json')
    evproof=read(EVIDENCE/'verification.json');attribution=read(ATTRIBUTION/'results.json');atproof=read(ATTRIBUTION/'verification.json')
    require(proof['passed'] and evproof['passed'] and atproof['passed'],'验收尚未通过')
    for source in (evproof,proof,atproof):
        for rel,h in {**source.get('input_hashes',{}),**source.get('artifact_hashes',{})}.items():
            require(digest(BASE/rel)==h,'生成报告前证据指纹不同')
    sys.path.append('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    matplotlib.rcParams['font.family']=font.get_name();matplotlib.rcParams['axes.unicode_minus']=False
    cases=list(attribution['cases']);labels=[NAMES[k.split('__')[0]].replace('（主方案）','') for k in cases]
    fig,axes=plt.subplots(1,2,figsize=(14,6),gridspec_kw={'width_ratios':[1.55,1]})
    colors=['#71869c','#d57957','#2b978d','#c2a13d','#9574a3']
    y=np.arange(len(cases));negative=np.zeros(len(y));positive=np.zeros(len(y))
    for (part,label),color in zip(PARTS.items(),colors):
        values=np.array([attribution['cases'][k]['profit']['cagr_gap_allocation_by_log_link'][part]*100 for k in cases])
        left=np.where(values>=0,positive,negative)
        axes[0].barh(y,values,left=left,height=.58,label=label,color=color)
        positive+=np.maximum(values,0);negative+=np.minimum(values,0)
    axes[0].set_yticks(y,labels);axes[0].invert_yaxis();axes[0].axvline(0,color='#444',lw=.8)
    axes[0].set_xlabel('与人民币SPY的CAGR差额分配（百分点，按对数链接）')
    axes[0].set_title('未投资资金是主要收益差额来源',pad=15)
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=3,frameon=False,fontsize=9)
    bridge=attribution['cases']['group4_risk_parity__base']['cash_bridge']['mean_components']
    blabels=['永久现金','月初空槽位','月内移除目标','仓位与风险上限','漂移带','每日换手预算','目标与现仓差']
    values=np.array(list(bridge.values()))*100
    axes[1].barh(np.arange(7),values,color=['#71869c' if v>=0 else '#2b978d' for v in values],height=.58)
    axes[1].set_yticks(np.arange(7),blabels);axes[1].invert_yaxis();axes[1].axvline(0,color='#444',lw=.8)
    axes[1].set_xlim(-3,23)
    for i,v in enumerate(values): axes[1].text(v+.4 if v>=0 else .4,i,f'{v:.2f}',va='center',ha='left',fontsize=9)
    axes[1].set_title('主方案平均现金比例分解',pad=15);axes[1].set_xlabel('可调配资产的比例（百分点，分母不含应收）')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('固定30只ETF分组策略：收益差距归因',fontproperties=font,fontsize=17)
    fig.text(.5,.035,'2016-09-12 至 2026-09-11；已验收旧基础情景的诊断。机会成本不等于可执行的加仓收益。',ha='center',fontproperties=font,fontsize=10,color='#555')
    fig.subplots_adjust(left=.16,right=.97,top=.84,bottom=.25,wspace=.4)
    figure=ATTRIBUTION/'return_attribution.png';fig.savefig(figure,dpi=160,facecolor='white');plt.close(fig)

    bench=replay['cases']['fixed_control__base']['metrics']['spy_total_cny']
    rows=[]
    for key,item in replay['cases'].items():
        if item['kind']=='event': continue
        variant,scenario=key.split('__');m=item['metrics']['strategy']
        rows.append(f"| {NAMES[variant]} | {SCENARIOS[scenario]} | {m['cagr']:.2%} | {m['mean_daily_drawdown']:.2%} | {m['max_drawdown']:.2%} |")
    sources=[]
    for item in evidence['events']:
        e=item['event'];source=item['source'];note='年份冲突，保留额外延迟' if not item['payment_semantics_verified'] else '指定交易投资者发放日可领取'
        if item['scope']=='warmup_signal_only': note+='；仅预热信号'
        rel=(EVIDENCE/source['file']).relative_to(BASE).as_posix()
        sources.append(f"| [{e['symbol']}](../{rel}) | {e['cash_per_share_decimal']} | {e['record_date']} | {e['ex_date']} | {e['announced_payment_date']} | {note} |")
    report=f'''# 剩余分红原件与支付模型影响验收

日期：2026-09-14。33份原件和111个影响情景已完成独立核验；工程验收通过，**投资准入仍未通过**。18个完整十年情景没有任何一个满足收益、逐日平均回撤及20%最大回撤联合门槛。

## 原件核验结果

取得剩余31项区间分红及2项预热分红的发行人PDF，交易所索引共33个不筛标题的事件窗口、33页响应。首次标题匹配下载24份，另9份“收益分配公告”从同一原始索引补取；首次零候选记录保留。88个物理页全部渲染，66个包含金额、日期、权益或领取条款的完整页面实际目视核对。其余22页为封面，没有计入目视页数。

主提取器使用pypdf；独立复核使用pdfplumber/pdfminer并按表格单元格读取金额，以避免换行单位与金额错序。33项金额和除息日期均与旧值一致，31项区间登记/发放日期表也一致；2项预热分红补齐登记和支付日期。8组测试覆盖错误代码、管理人、单位、金额、日期顺序、场内外歧义、跨页条款和领取年份冲突。

32份领取说明与日期表一致。**SH510880的2017年公告表格写2017-01-26发放，正文领取句写2016-01-26。** 两处原文均保留，支付语义仍标记为待核实，保守额外延迟一个交易日。没有自行更正年份，也没有把公告计划日期标成券商实际到账。

已知69项行动（含14项拆分、53项区间分红、2项预热分红）的金额/日期证据现均来自原件；这不等于全历史行动、存续和清盘目录已完整，也不等于所有行动的支付语义或实际到账已确认。

## 支付模型影响与验收

预登记目录：`{directory.relative_to(BASE).as_posix()}`，共1022项输入指纹。仅生成新行动版本，30项区间分红取消旧第三方日程的额外一日延迟，2017冲突继续保留；其余成本、资格、仓位和调仓规则不变。公告支付在收盘后入模型现金，不能在当日开盘提前使用。

64,238行总收益信号与旧版逐字段完全相等，全部30只、69项行动一致。93个事件窗口（31项×3压力）、15个分组策略情景及3个固定控制全部完成。独立重建45,870个日账本行，核查信号、整手、现金、应收、费用与30%共享换手预算，31项分红都有非零权益。

30项事件有非零模型支付；2026-09-09有色ETF分红公告发放日为2026-09-14，超过历史行情截止2026-09-11，继续保留应收。不能用当前自然日期补造历史行情或支付观察。模型回放不计真实前瞻，合格真实前瞻仍为0/126。

## 全部完整十年情景

2016-09-12—2026-09-11，共2428个中国交易日，100万元现金开始。策略扣除费用；基准为同中国交易日、人民币计价的SPY含分红。逐日平均回撤为相对历史净值高点的损失均值，新高日计零；年度最大回撤均值仅作补充。该定义用于持续持有压力评价，并非声称业内只有一种“平均回撤”定义。

| 方法 | 情景 | CAGR | 逐日平均回撤 | 最大回撤 |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(rows)}
| 人民币SPY含分红 | 共同基准 | {bench['cagr']:.2%} | {bench['mean_daily_drawdown']:.2%} | {bench['max_drawdown']:.2%} |

三个固定控制的成交、现金和净值与旧原件执行版完全一致。分组方法有少量成交与净值变化，方向不一致；支付时点修正没有解决其收益不足。不能把个别延迟情景上涨解释为应主动延迟分红到账。

## 下一步与不通过项

收益损失归因见[分组策略收益归因](SP500_GROUP_RETURN_ATTRIBUTION.md)。下一项研究应先隔离“风险预算被仓位上限截断后直接留现金”的资金利用机制，在保留单只/组别/总仓位和波动约束下验证可行资金重新分配，不根据已经失败的参数邻域挑一个结果宣称样本外达标。

仍缺满足联合指标的策略、全历史完整目录、账户核对与126日真实前瞻。当前正式指针仍为`20260912-220345-f20ea3f5`；每日自动化和冻结候选未切换，不下实盘订单。

## 逐项原件

金额单位为元/份；原公告单位为元/10份。链接指向本地原始PDF，索引URL、下载时间及哈希见证据目录。

| 原件/代码 | 每份现金 | 登记日 | 除息日 | 公告发放日 | 说明 |
| --- | ---: | --- | --- | --- | --- |
{chr(10).join(sources)}

复核已落盘回放：`.venv\\Scripts\\python.exe -m research.goal30.run_dividend_payment_replay verify --directory {directory.relative_to(BASE).as_posix()}`。
原件验收文件：`{EVIDENCE.relative_to(BASE).as_posix()}/verification.json`。该验收入口禁止覆盖已验收版本。
'''
    path=BASE/'docs/SP500_REMAINING_DIVIDENDS_ACCEPTANCE.md';path.write_text(report,encoding='utf-8')
    table=[]
    for key,item in attribution['cases'].items():
        p=item['profit'];parts=p['cagr_gap_allocation_by_log_link']
        table.append('| '+NAMES[key.split('__')[0]]+f" | {p['strategy_cagr']:.2%} | {p['cagr_gap']*100:+.2f} | "+
                     ' | '.join(f'{parts[k]*100:+.2f}' for k in PARTS)+' |')
    p=attribution['cases']['group4_risk_parity__base']['profit'];b=attribution['cases']['group4_risk_parity__base']['cash_bridge']
    report=f'''# 分组策略收益损失归因

日期：2026-09-14。对上一轮冻结的五个基础情景逐日拆分，未产生新策略。主方案CAGR为{p['strategy_cagr']:.2%}，与人民币SPY相差{p['cagr_gap']*100:.2f}个百分点。主要差额来自未投资资金的基准机会成本，持仓相对表现也为负；仅修正分红支付日期不能解决这两项问题。

## 收益差额

以下贡献按逐日对数链接，再用共同缩放系数分配CAGR差额，可精确相加。它是已发生路径的描述性归因，分配方式不是唯一的因果解释；现金机会成本不表示池外SPY可以买入，也不是“将现金全部加仓后可以获得的收益”。

| 方法 | CAGR | 对基准差额 | 未投资资金 | 持仓相对表现 | 交易时点 | 佣金 | 滑点/冲击 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

后六列单位均为百分点。未投资资金包括现金及分红应收，按前一日余额计算基准机会成本。持仓项含拆分调整后的价格损益、除息应收及份额取整差，再扣除同等期初市值的基准收益。交易时点项为当日开盘成交至收盘的增减仓损益；费用独立扣除。应收转现金不重复计收益。

![收益与现金比例归因](../{figure.relative_to(BASE).as_posix()})

## 为什么经常留现金

主方案平均可调配资产现金比例为{b['mean_liquid_cash_fraction']:.2%}，分母为现金加ETF市值，不含应收。逐日目标桥可拆为：

| 项目 | 平均比例贡献 |
| --- | ---: |
| 固定现金储备 | {bridge['permanent_cash_reserve']:.2%} |
| 月初未填满的槽位 | {bridge['unfilled_slots_at_monthly_review']:.2%} |
| 月内退出后移除的目标 | {bridge['removed_targets_since_monthly_review']:.2%} |
| 单只、组别、总仓位及波动上限缩减 | {bridge['risk_and_position_caps']:.2%} |
| 漂移带的净影响 | {bridge['drift_band_effect']:.2%} |
| 每日换手预算 | {bridge['daily_turnover_budget']:.2%} |
| 次日目标与当日实际持仓之差 | {bridge['target_to_current_holdings_gap']:.2%} |

这是比例恒等式，不可把各项直接当成收益损失。最后一项允许为负，例如待卖出的实际持仓仍高于次日目标。主方案有{b['days_with_monthly_targets_removed']}个信号日存在月内目标移除，原月度目标已重新通过趋势/资格但仍等待下月重选的记录为{b['recovered_original_target_symbol_days_waiting_next_month']}个“标的×日”。

## 历史连续性和缺价

2427个信号日逐一检查固定30只，共72,810个“标的×日”，互斥分类为：未上市10,354；初始253日观察窗口不足5,800；完整历史但趋势不合格26,896；正向趋势27,749；窗口含缺价且与已知拆分窗口重合1,012；其余含缺价999。其中同时通过当日资格的可用信号23,735，平均每日9.78只。

区间内上市后的原始价格共缺8个“标的×日”，其中4个与已知拆分登记至生效窗口重合。滚动253日要求将这些少数缺口扩展为2,011个受影响的“标的×日”。这不能证明全部缺口都是停牌，也不能把滚动影响次数当成独立停牌天数。明细见`output/goal30/group_attribution/20260914/missing_quotes.csv`。

数据起点与253条连续价格要求共同造成初期等待，不能删去现金期或补造上市前价格。若后续修复缺口，必须取得对应真实行情或停牌原件，再决定信号规则；不能插值后称真实价格完整。

## 验收和机制决策

4组手工算例通过，覆盖拆分价格跳变中性、分红仅计一次、取整、买卖方向和跨期链接。独立核验从冻结持仓、未复权原价及成交价重建12,140个日账本行，并直接重算72,810个候选历史分类；最大数值差{atproof['maximum_numeric_difference']:.3g}。逐日经济净值、费用、年度和全期对数增长贡献及现金比例桥均闭合。

下一阶段先检验受约束的资金重新分配：风险均衡权重被35%单只上限截断后，若其他当时合格标的仍有风险和仓位余量，能否在原约束内重新配置。保持原趋势、资格、月度时点和费用作为对照，先隔离这一机制。初期数据不足与月内重入是另外的问题，不能同时改动后将效果归因于单一原因。该研究仍使用已看历史，不产生新的样本外证据，也不预设会达标。

验收源目录：`output/goal30/group_attribution/20260914`。逐日金额归因、现金比例桥、缺价明细、输入哈希及独立验收均已保存。复核：`.venv\\Scripts\\python.exe -m research.goal30.audit_group_attribution`。
'''
    attribution_doc=BASE/'docs/SP500_GROUP_RETURN_ATTRIBUTION.md';attribution_doc.write_text(report,encoding='utf-8')
    report_files=[path,attribution_doc,figure,Path(__file__)]
    write_json_atomic(ATTRIBUTION/'report_manifest.json',{'source_hashes':{
        (directory/'verification.json').relative_to(BASE).as_posix():digest(directory/'verification.json'),
        (ATTRIBUTION/'verification.json').relative_to(BASE).as_posix():digest(ATTRIBUTION/'verification.json'),
        (EVIDENCE/'verification.json').relative_to(BASE).as_posix():digest(EVIDENCE/'verification.json')},
        'report_hashes':{p.relative_to(BASE).as_posix():digest(p) for p in report_files},'figure_visually_reviewed':False})
    print(json.dumps({'reports':[p.relative_to(BASE).as_posix() for p in report_files[:3]]}))


if __name__=='__main__': create()
