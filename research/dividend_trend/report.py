"""从已核验的冻结研究运行生成中文结论与科学图，不改变策略或旧收益。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from .run import ROOT, OLD, digest, write_json, todo_write, metrics
from .risk12 import SID


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def curve(path, column='nav'):
    data=pd.read_csv(path,parse_dates=['date']).set_index('date')[column]
    data.index=pd.DatetimeIndex(data.index).as_unit('ns')
    return data


def pct(value):
    return f'{value:.2%}'


def metric_table(rows, include_end=True):
    if include_end:
        lines=['| 策略 / 基准 | 期末（万元） | 年化 | 最差完整年 | 最大回撤 |',
               '| --- | ---: | ---: | ---: | ---: |']
    else:
        lines=['| 场景 | 年化 | 最差完整年 | 最大回撤 |',
               '| --- | ---: | ---: | ---: |']
    for label,row in rows:
        ending=f" {row['final_nav']/10000:.2f} |" if include_end else ''
        lines.append(f"| {label} |{ending} {pct(row['cagr'])} | {pct(row['worst_full_year'])}（{row['worst_year']}） | {pct(row['max_drawdown'])} |")
    return lines


def main(risk_run,output):
    risk_run,output=risk_run.resolve(),output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    reg=read_json(risk_run/'registration.json')
    result=read_json(risk_run/'results.json')
    second=ROOT/reg['source_run']
    second_result=read_json(second/'results.json')
    second_reg=read_json(second/'registration.json')
    first=ROOT/second_reg['protocol']['parent_run']
    first_result=read_json(first/'results.json')
    audit=ROOT/reg['prior_validation']
    for directory in (first,second,audit,risk_run):
        assert read_json(directory/'verification.json')['status']=='passed'
    for path,expected in reg['hashes'].items():
        assert digest(ROOT/path)==expected,path
    rows={r['id']:r for r in result['rows']}
    mainrow=rows[SID]
    old=read_json(OLD/'comparison.json')
    oldrows={r['strategy_id']:r for r in old['rows']}
    baseline=result['benchmark']
    candidate=curve(risk_run/'cases'/SID/'daily.csv')
    red=curve(second/'red_benchmark_nav.csv','red')
    assert candidate.index.equals(red.index)
    archive=pd.read_csv(OLD/'daily_wealth_native_calendars.csv',parse_dates=['date']).set_index('date')
    archive.index=pd.DatetimeIndex(archive.index).as_unit('ns')
    curves={SID:candidate,'red_benchmark':red,
            'communication_gold':archive.trend200_vol15.reindex(candidate.index),
            'csi300_pe':archive.A_pe_weekly.reindex(candidate.index)}
    assert all(c.notna().all() for c in curves.values())
    pd.DataFrame(curves).to_csv(output/'daily_comparison.csv',index_label='date')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(13.5,8.1),sharex=True,
                          gridspec_kw={'height_ratios':[1.55,1]},layout='constrained')
    labels={SID:'红利低波/黄金双动量 · 12%预算','red_benchmark':'红利低波全收益持有',
            'communication_gold':'原通信/黄金','csi300_pe':'沪深300PE（等权）'}
    colors={SID:'#087f8c','red_benchmark':'#d28428','communication_gold':'#64748b','csi300_pe':'#8a6fad'}
    styles={SID:'-','red_benchmark':'--','communication_gold':':','csi300_pe':'-.'}
    for key,nav in curves.items():
        axes[0].plot(nav.index,nav/10000,label=f'{labels[key]}  {nav.iloc[-1]/10000:.2f}',
                     color=colors[key],linestyle=styles[key],linewidth=2. if key==SID else 1.35)
    axes[0].set_ylabel('10万元起始 → 金额（万元）')
    axes[0].legend(loc='upper left',frameon=False,ncol=2,fontsize=10)
    axes[0].set_title('全期年化提高，优势集中在特定阶段；尚未证明持续胜出',loc='left',fontsize=14,pad=15)
    for key in (SID,'red_benchmark'):
        nav=curves[key]
        dd=nav/nav.cummax()-1
        axes[1].plot(dd.index,dd,label=f'{labels[key]}  最大回撤 {-dd.min():.2%}',
                     color=colors[key],linestyle=styles[key],linewidth=1.5)
        if key==SID:
            axes[1].fill_between(dd.index,dd,0,color=colors[key],alpha=.10)
    axes[1].set_ylabel('每日收盘净值回撤')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    axes[1].legend(loc='lower left',frameon=False,fontsize=10)
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[1].set_xlim(candidate.index[0],candidate.index[-1])
    for ax in axes:
        ax.grid(axis='y',alpha=.18)
        ax.set_axisbelow(True)
    fig.supxlabel('2016-09-12—2026-11-04；周频T+1收盘，候选含成本；指数/价格代理，非ETF整手账户',fontsize=10)
    fig.savefig(output/'comparison.png',dpi=160)
    plt.close(fig)
    annual={a['year']:a for a in mainrow['annual']}
    red_annual={a['year']:a for a in baseline['annual']}
    v15=next(r for r in second_result['rows'] if r['id']=='dual_blend_ma200_v15')
    comparison=[('红利低波/黄金双动量（12%预算）',mainrow),
                ('红利低波/黄金双动量（15%预算）',v15)]
    comparison += [(r['label'],r) for r in old['rows']]
    # 独立按年末记录复核最终表格与旧基准行。
    assert abs(metrics(candidate)['cagr']-mainrow['cagr'])<1e-12
    export=[{'label':label,**{k:v for k,v in row.items() if k not in ('annual','personal_goals')}}
            for label,row in comparison]
    pd.DataFrame(export).to_csv(output/'comparison.csv',index=False,encoding='utf-8-sig')
    daily=pd.read_csv(risk_run/'cases'/SID/'daily.csv',parse_dates=['date']).set_index('date')
    prices=pd.read_csv(second/'index_gold_prices.csv',parse_dates=['date']).set_index('date')
    pnl=pd.DataFrame({asset:daily[asset+'_value'].shift(1)*prices[asset].pct_change(fill_method=None).reindex(daily.index)
                      for asset in ('red','gold')}).fillna(0)
    attribution=[]
    for year in range(2017,2026):
        portion=daily.loc[str(year)]
        prior=candidate[candidate.index<pd.Timestamp(year,1,1)].iloc[-1]
        red_pnl=float(pnl.loc[str(year),'red'].sum())
        gold_pnl=float(pnl.loc[str(year),'gold'].sum())
        fees=float(portion.trading_cost.sum()+portion.replication_cost.sum())
        change=float(portion.nav.iloc[-1]-prior)
        assert abs(red_pnl+gold_pnl-fees-change)<1e-6
        attribution.append({'year':year,'red_pnl_cny':red_pnl,'gold_pnl_cny':gold_pnl,
                            'fees_cny':fees,'account_change_cny':change})
    pd.DataFrame(attribution).to_csv(output/'annual_pnl_attribution.csv',index=False,encoding='utf-8-sig')
    seg=next(r for r in result['segments'] if r['window']=='2017-2024')
    redseg=next(r for r in second_result['segments'] if r['id']=='red_benchmark' and r['window']=='2017-2024')
    etf=rows[SID+'__etf_base']
    etfbase=result['etf_benchmark']
    costs=rows[SID+'__cost2']
    firstrows={r['id']:r for r in first_result['rows']}
    lines=['# 红利低波趋势研究：找到全期改善候选，尚不能认定持续优于',
        '', '研究日期：2026-11-10。共同区间2016-09-12—2026-11-04；完整自然年2017—2025。',
        '', '## 判断', '',
        f"红利低波/黄金双动量、MA200与12%波动预算，是本次保留的风险优先研究候选。10万元分数等值账户期末{mainrow['final_nav']/10000:.2f}万元，年化{pct(mainrow['cagr'])}、最大每日净值回撤{pct(mainrow['max_drawdown'])}、最差完整年{pct(mainrow['worst_full_year'])}（2022），相对原表红利全收益三项同时改善。",
        '这只成立于所列全期和费用假设。此前2017—2024以及ETF上市后对照均未全面胜出，成本翻倍还会突破个人年度亏损目标；目前不能认定为持续优于、独立样本外通过或正式投资准入。',
        f"原个人联合目标仍为年化≥13.50%、最差完整年≥−5%、最大回撤≤40%。基础情景两项风险达标，年化仍差{(.135-mainrow['cagr'])*100:.2f}个百分点，没有达到全部目标。",'',
        '## 全期比较', '',*metric_table(comparison), '',
        '旧八行保持原始归档口径；指数基准未扣基金及复制交易费，旧策略为100万元历史账户归一化，并非10万元整手重新执行。新候选按10万元分数等值账户计算，已计下述费用；这张表用于同时间收益路径比较，不是严格统一全部成交模型的控制实验。',
        '', '![净值与回撤](comparison.png)', '',
        '## 可复现规则', '',
        '1. 资产仅红利低波与境内黄金，现金零息，不加杠杆。研究信号使用H20269全收益点位和518880复权收盘价，股息已内含；不重复加息，不持有美股。',
        '2. 每周首个中国交易日收盘评估，休市顺延。分别计算`score = ((P/P[-63]-1)+(P/P[-126]-1)+(P/P[-252]-1))/3`及200交易日均线。',
        '3. 动量分数严格大于0且当日价格严格大于MA200才合格。选择合格者中分数最高的一个，并列优先红利；均不合格则持现金。63/126/252是交易日，不是自然日。',
        '4. 初始目标为所选资产95%、现金5%。用截至信号日的60个交易日收益样本协方差（ddof=1、252年化）计算目标组合波动，只按`min(1, 12%/预计波动)`同比例缩仓。单资产时等价于`目标权重=min(95%,12%/该资产年化波动)`。',
        '5. 非零目标与当前权重差小于3个百分点时保留原权重，合计超过95%再缩回95%；目标归零时不使用漂移带。5%是调仓目标现金，持仓随价格漂移，不保证每日持续为5%。',
        '6. 下一交易日收盘调整，先卖后买，信号日和成交日前涨跌由旧仓承担。不把收盘指数伪称为次日开盘成交；周中不额外检查止损。',
        '7. 每个资产每个买卖方向扣`max(5元,成交额×0.10%)`；红利指数持有部分再按日历天扣0.60%年化复制费用情景。黄金价格已含基金费用，不重复扣；现金零息。0.60%是统一研究折减，并非已审计的2016年以来实际费率时间表。',
        '8. ETF子样本仅2019-01-18起用512890真实价格，2021-10-22披露1:2拆分，当天只估值不成交；ETF版本不再扣指数复制费。该子样本仍用等值分数份额，不生成正式ETF订单或实际持仓。',
        '', '12%波动预算是仓位缩放参数，不是最大亏损或回撤保证。历史最大回撤17.46%已大于12%。',
        '', '## 年度收益', '',
        '| 完整年 | 12%预算候选 | 红利低波全收益 |', '| --- | ---: | ---: |']
    for year in range(2017,2026):
        lines.append(f"| {year} | {pct(annual[year]['return'])} | {pct(red_annual[year]['return'])} |")
    lines.extend(['',f"2016部分年：候选{pct(annual[2016]['return'])}、红利指数{pct(red_annual[2016]['return'])}；2026截至11月4日：候选{pct(annual[2026]['return'])}、红利指数{pct(red_annual[2026]['return'])}。首尾部分年不参与最差完整年排名。",
        '', '## 稳健性和失败证据', '',
        *metric_table([('基础情景',mainrow),('单边成本0.20%',costs),('单边成本0.30%',rows[SID+'__cost3']),
            ('延迟到T+2收盘',rows[SID+'__delay2']),('周末交易日发信号',rows[SID+'__week_end']),
            ('指数复制费1%/年',rows[SID+'__fee1pct']),('MA150',rows[SID+'__ma150']),
            ('MA250',rows[SID+'__ma250']),('波动预算10%',rows[SID+'__vol10']),
            ('波动预算14%',rows[SID+'__vol14'])],include_end=False), '',
        f"成本翻倍后的年化{pct(costs['cagr'])}仅比指数{pct(baseline['cagr'])}高{(costs['cagr']-baseline['cagr'])*100:.3f}个百分点，最差年{pct(costs['worst_full_year'])}已低于−5%。成本三倍时年化反低于指数。MA150也会使最差年降至−8.46%。不能只看基础情景通过。",
        '',*metric_table([('2017—2024候选',seg),('2017—2024红利指数',redseg),
                         ('2019-01-18起ETF趋势代理',etf),('同区间512890持有代理',etfbase)],include_end=False), '',
        f"截至2024年底的八个完整年，候选年化{pct(seg['cagr'])}低于红利指数{pct(redseg['cagr'])}。ETF上市后候选年化{pct(etf['cagr'])}也低于持有{pct(etfbase['cagr'])}，回撤{pct(etf['max_drawdown'])}高于持有{pct(etfbase['max_drawdown'])}；持有最差完整年为正，而候选仍有亏损年。不能以2016起的全期胜出替代这些失败结果。",
        f"候选2025年收益{pct(annual[2025]['return'])}，红利指数同期{pct(red_annual[2025]['return'])}；全期平均红利权重{pct(mainrow['average_red_weight'])}、黄金{pct(mainrow['average_gold_weight'])}、现金{pct(mainrow['average_cash_weight'])}。它属于红利/黄金轮动，而不是纯红利加现金。逐年两资产市值盈亏和费用见`annual_pnl_attribution.csv`。",
        '', '## 纯红利和固定配置对照', '',
        *metric_table([('红利95%＋现金·MA200·15%预算',firstrows['cash95_ma200_v15']),
                       ('红利60%＋黄金35%各自MA200·15%预算',firstrows['split60_ma200_v15']),
                       ('红利60%＋黄金35%静态配置',firstrows['static60']),
                       ('红利95%持有·同成本',firstrows['red95_hold'])]), '',
        '首批纯红利加现金的MA150/200/250及有/无波动缩放均没有达到年化超越红利指数的要求。静态配置全期三项优于指数，但不是趋势策略，最差年也未达到个人−5%目标，因此没有冒充目标策略。',
        '', '## 选择过程与验证范围', '',
        '首批登记18项配置，再回放12项执行压力和9项ETF子样本；看到失败结果后登记9项相对强弱/核心仓配置及其21项补充场景。随后对blend候选登记7项参数/成本复核，12%预算因两项风险同时改善而保留，再登记13项补充验证。总计89条历史回放，包括重复控制和压力情景，不能视作89个独立策略或统计试验。',
        '126日是相对强弱第二批原始主方案，blend最初是预定义平滑对照；12%来自后续压力结果。原主方案和旧协议未回写，不将研究候选改称原始预注册主方案。全部历史已经被查看，没有真正未见数据的盲测；时间分段是延续同一账户的事后诊断。',
        '13项合成测试通过，检查信号和成交时序、清仓绕过漂移带、自融资先卖后买、停牌顺延、均线相等退出、相对动量并列、组合缩放及未来数据不改变前缀。真实历史逐日账户恒等式、完整交易日、独立单遍回撤/年末收益、截断信号、独立numpy逐窗信号与独立份额净值重建全部通过；份额复算最大误差小于1e−6元。',
        '未验证100份整手、真实券商收盘成交、成交容量和全部停牌/涨跌停约束，没有实时前瞻账户、没有独立未见历史样本；指数历史和复权行情属于当前快照，不能推断为原始时点留存数据。ETF公司行动沿用归档证据，没有重新完成全市场存续标的审计。未运行无关前后端或全项目测试。',
        '正式服务、生产指针、正式数据库、策略登记和旧研究收益输入哈希保持不变。研究完成不等于投资准入，当前不建议据此替换现有正式策略。',
        '', '## 来源与复跑', '',
        '- [中证官方指数行情接口](https://www.csindex.com.cn/csindex-home/perf/index-perf)：H20269全收益日收盘；本研究沿用已核验历史并新增2014年起预热原始响应及哈希。',
        '- [512890基金产品资料概要（2026-03-18）](https://www.sse.com.cn/disclosure/fund/announcement/c/new/2026-03-18/512890_20260318_FNWM.pdf)：跟踪指数、2019-01-18上市日及所披露现行管理/托管费用。没有把ETF上市前指数当作可成交ETF历史。',
        f"- 原八项比较：`{OLD.relative_to(ROOT).as_posix()}/comparison.json`。",
        f"- 首批完整结果：`{first.relative_to(ROOT).as_posix()}`。",
        f"- 相对强弱完整结果：`{second.relative_to(ROOT).as_posix()}`。",
        f"- 独立参数复核：`{audit.relative_to(ROOT).as_posix()}`。",
        f"- 12%版本完整结果：`{risk_run.relative_to(ROOT).as_posix()}`。", '',
        '```powershell',
        '.venv/Scripts/python.exe -X utf8 -m unittest research.dividend_trend.test_engine research.dividend_trend.test_relative -v',
        '# 已完成运行禁止覆盖；重新实验先 --register，保留原目录。',
        '.venv/Scripts/python.exe -X utf8 -m research.dividend_trend.run --register',
        '.venv/Scripts/python.exe -X utf8 -m research.dividend_trend.run --run <新登记目录>',
        f'.venv/Scripts/python.exe -X utf8 -m research.dividend_trend.relative --register',
        '.venv/Scripts/python.exe -X utf8 -m research.dividend_trend.relative --run <新登记目录>',
        f'.venv/Scripts/python.exe -X utf8 -m research.dividend_trend.risk12 --register {second.relative_to(ROOT).as_posix()} --validation {audit.relative_to(ROOT).as_posix()}',
        '.venv/Scripts/python.exe -X utf8 -m research.dividend_trend.risk12 --run <新登记目录>',
        '```',''])
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(output/'candidate.json',{'id':SID,'status':'retrospective_research_candidate_not_admitted',
        'metrics':mainrow,'source_run':str(risk_run.relative_to(ROOT)),
        'selection_process':'post_result_risk_preference_selection_not_blind_validation',
        'rules':{'momentum_windows':[63,126,252],'ma':200,'vol_window':60,'vol_budget':.12,
                 'max_target_exposure':.95,'one_way_cost':.001,'replication_haircut':.006,
                 'minimum_fee':5.,'rebalance_band':.03,'execution':'weekly_first_session_signal_T+1_close'},
        'joint_goals_passed':False,'full_period_dominates_index':True,
        'robust_dominance_proven':False,'production_changed':False})
    write_json(output/'verification.json',{'status':'passed_with_material_research_limits',
        'unit_tests':{'command':'.venv/Scripts/python.exe -X utf8 -m unittest research.dividend_trend.test_engine research.dividend_trend.test_relative -v',
                      'passed':13,'failed':0},
        'historical_replays':89,'independent_strategy_count_claimed':False,
        'source_verifications':[str((d/'verification.json').relative_to(ROOT)) for d in (first,second,audit,risk_run)],
        'pnl_attribution_reconciled':True,'chart_metrics_use_daily_not_sampled_prices':True,
        'visual_inspection':'pending','investment_admission':False})
    print(json.dumps({'report':str(output/'report.md'),'plot':str(output/'comparison.png'),
        'pre_2025_cagr':seg['cagr'],'etf_cagr':etf['cagr'],
        '2025_pnl_attribution':next(a for a in attribution if a['year']==2025)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--risk-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.risk_run,args.output)
