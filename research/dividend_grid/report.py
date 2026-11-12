"""从冻结回放生成中文研究说明、贡献诊断与可分享科学图。"""
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

from research.dividend_trend.run import ROOT,digest,write_json,metrics
from .data import frame_read,load
from .engine import simulate
from .signals import targets
from .run import todo_write

LABELS={'hold95_cash':'95%红利持有·同成本','static60_cash':'60%红利＋现金',
        'static60_bond':'60%红利＋35%国债','grid_cash':'底仓＋网格＋现金',
        'grid_bond':'底仓＋网格＋国债','risk60_cash':'底仓＋风控＋现金',
        'risk60_bond':'底仓＋风控＋国债','grid_risk_cash':'底仓＋网格＋风控＋现金',
        'grid_risk_bond':'主方案：底仓＋网格＋风控＋国债',
        'grid_risk_bond_core50':'底仓50%', 'grid_risk_bond_ma180':'网格MA180',
        'grid_risk_bond_ma250':'网格MA250','grid_risk_bond_narrow':'较窄网格',
        'grid_risk_bond_wide':'较宽网格','grid_risk_bond_vol18':'风险波动阈值18%／16%',
        'grid_risk_bond_vol22':'风险波动阈值22%／20%','grid_risk_bond_recover1':'恢复确认1周',
        'grid_risk_bond_recover3':'恢复确认3周','grid_risk_bond_volgrid':'波动率调整网格',
        'relative_risk_bond':'40日相对收益网格'}


def table(items,with_end=True):
    header='| 配置 | '+('期末万元 | ' if with_end else '')+'年化 | 最差完整年 | 最大回撤 |'
    lines=[header,'| --- | '+('---: | ' if with_end else '')+'---: | ---: | ---: |']
    for name,row in items:
        final=f"{row['final_nav']/10000:.2f} | " if with_end else ''
        lines.append(f"| {name} | {final}{row['cagr']:.2%} | {row['worst_full_year']:.2%}（{row['worst_year']}） | {row['max_drawdown']:.2%} |")
    return lines


def build(directory,output):
    directory,output=directory.resolve(),output.resolve()
    assert not (output/'completion.json').exists(),'已完成报告不覆盖'
    output.mkdir(parents=True,exist_ok=True)
    reg=json.loads((directory/'registration.json').read_text(encoding='utf-8'))
    result=json.loads((directory/'results.json').read_text(encoding='utf-8'))
    verification=json.loads((directory/'verification.json').read_text(encoding='utf-8'))
    assert verification['status']=='passed'
    for name,h in reg['protected_hashes'].items():assert digest(ROOT/name)==h,name
    proxy,etf,actions=load(ROOT/reg['data_directory'])
    rows={r['id']:r for r in result['rows']}
    main=rows['grid_risk_bond__proxy'];live=rows['grid_risk_bond__etf']
    primary=next(c for c in reg['candidates'] if c['id']=='grid_risk_bond')
    # 再用截断的行情、信号和账户检验真实历史账户前缀，防止回放读取未来。
    prefixes={}
    for mode,market,start in [('proxy',proxy,'2016-09-12'),('etf',etf,'2019-01-18')]:
        sig=targets(proxy.loc[:'2021-12-31'],primary,start)
        account,_,_,_=simulate(market.loc[:'2021-12-31'],sig,start,actions,etf=mode=='etf')
        full=frame_read(directory/'cases'/f'grid_risk_bond__{mode}'/'daily.csv')
        error=float((account.nav-full.nav.loc[account.index]).abs().max())
        assert error<1e-7,error
        prefixes[mode]=error
    write_json(output/'additional_verification.json',{'status':'passed','historical_account_prefix_errors':prefixes,
         'notice_visual_review':'2026年9月分红公告第1/2页已渲染检查：每份1.2747元，场内登记9月17日，除息9月18日，到账9月23日；场外日期不用于账本',
         'pdf_page_hashes':{p.name:digest(p) for p in (ROOT/reg['data_directory']/'raw').glob('bond_notice_page-*.png')},
         'scope':'原研究之外的补充验证；不修改冻结原件'})
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                        'axes.unicode_minus':False,'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False})
    styles={'passive':('#cf8b2c','--'),'static60_bond':('#7d6ba6',':'),
            'grid_bond':('#718296','-.'),'grid_risk_bond':('#087f8c','-')}
    plots=[]
    for mode in ('proxy','etf'):
        fig,axes=plt.subplots(2,1,figsize=(12.8,7.7),sharex=True,layout='constrained',
                             gridspec_kw={'height_ratios':[1.5,1]})
        passive=frame_read(directory/f'{mode}_passive.csv').iloc[:,0]
        series={'passive':passive}
        for name in ['static60_bond','grid_bond','grid_risk_bond']:
            series[name]=frame_read(directory/'cases'/f'{name}__{mode}'/'daily.csv').nav
        pd.DataFrame(series).to_csv(output/f'{mode}_daily_comparison.csv',index_label='date')
        for key,nav in series.items():
            color,style=styles[key];label=('红利低波全收益持有' if mode=='proxy' else '512890持有代理') if key=='passive' else LABELS[key].replace('主方案：','')
            axes[0].plot(nav.index,nav/10000,color=color,linestyle=style,linewidth=1.8 if key=='grid_risk_bond' else 1.25,
                         label=f'{label}  {nav.iloc[-1]/10000:.2f}')
            if key in ['passive','grid_risk_bond']:
                dd=nav/nav.cummax()-1
                axes[1].plot(dd.index,dd,color=color,linestyle=style,linewidth=1.4,
                             label=f'{label}  最大回撤 {-dd.min():.2%}')
        title=('2016起指数代理：减小回撤，年化收益仍落后持有' if mode=='proxy' else
               '2019起ETF整手账户：风险降低，同时付出明显收益成本')
        axes[0].set_title(title,loc='left',fontsize=14,pad=12)
        axes[0].set_ylabel('10万元起始 → 金额（万元）');axes[0].legend(loc='upper left',frameon=False,ncol=2,fontsize=9)
        axes[1].set_ylabel('每日收盘净值回撤');axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=1))
        axes[1].legend(loc='lower left',frameon=False,fontsize=9)
        axes[1].xaxis.set_major_locator(mdates.YearLocator(2));axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        for ax in axes:ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        fig.supxlabel('周首收盘信号、T+1收盘；候选含费用；债券上市前现金；ETF账户计整手、容量与分红到账',fontsize=9)
        path=output/f'{mode}_comparison.png';fig.savefig(path,dpi=160);plt.close(fig);plots.append(path)
    # 主账户按真实上一日份额分解年度股票、债券收益，分红应收归入债券市值收益。
    attribution=[]
    for mode,market in [('proxy',proxy),('etf',etf)]:
        d=frame_read(directory/'cases'/f'grid_risk_bond__{mode}'/'daily.csv')
        marks=market.reindex(d.index)
        red_before=d.red_quantity.shift(1).fillna(0)
        red_after=red_before.copy()
        if mode=='etf':red_after.loc['2021-10-22']*=2
        red_pnl=red_after*marks.red-red_before*marks.red.shift(1)
        bond_pnl=d.bond_quantity.shift(1).fillna(0)*marks.bond.diff()+d.dividend_accrued
        costs=d.trading_cost+d.replication_cost
        red_pnl=red_pnl.fillna(0);bond_pnl=bond_pnl.fillna(0)
        error=float((red_pnl+bond_pnl-costs-d.nav.diff().fillna(0)).abs().max())
        assert error<1e-7,error
        for year,group in d.groupby(d.index.year):
            select=group.index
            attribution.append({'mode':mode,'year':int(year),'red_pnl':float(red_pnl.loc[select].sum()),
                'bond_pnl':float(bond_pnl.loc[select].sum()),'costs':float(costs.loc[select].sum()),
                'nav_change':float((red_pnl+bond_pnl-costs).loc[select].sum())})
    pd.DataFrame(attribution).to_csv(output/'annual_pnl_attribution.csv',index=False,encoding='utf-8-sig')
    baseids=reg['protocol']['registered_cases'][:9]
    sensitivities=reg['protocol']['registered_cases'][9:]
    winners={mode:[r['candidate'] for r in result['rows'] if r['mode']==mode and r['id'].count('__')==1 and r['dominates_passive']]
             for mode in ['proxy','etf']}
    lines=['# 红利核心底仓＋逆向网格＋国债配置：实施与验证', '',
        '研究日期2026-11-11。固定历史截止2026-11-04，保留此前比较窗口；没有扩展成截至今天的新行情回测。', '',
        '## 结论', '',
        f"主方案全期年化{main['cagr']:.2%}、最差完整年{main['worst_full_year']:.2%}（{main['worst_year']}）、最大回撤{main['max_drawdown']:.2%}；同期红利全收益为{result['benchmarks']['proxy']['cagr']:.2%}、{result['benchmarks']['proxy']['worst_full_year']:.2%}、{result['benchmarks']['proxy']['max_drawdown']:.2%}。它实现了降回撤，但没有实现年化超越，且全期最差完整年仍低于−5%。",
        f"ETF共同上市后主方案年化{live['cagr']:.2%}、最差完整年{live['worst_full_year']:.2%}、回撤{live['max_drawdown']:.2%}。同区间512890持有代理分别为{result['benchmarks']['etf']['cagr']:.2%}、{result['benchmarks']['etf']['worst_full_year']:.2%}、{result['benchmarks']['etf']['max_drawdown']:.2%}。两项比较不能互相替换，也不能只展示更有利的窗口。",
        f"本批20个基础配置中，全期三项同时胜过指数的配置为{winners['proxy'] or '无'}；ETF子样本三项同时胜过持有的配置为{winners['etf'] or '无'}。所有57条回放保留，包括20×2个基础case及17条压力/执行对照；不是57个独立策略。",
        '原个人13.5%年化、最差完整年≥−5%、回撤≤40%的联合目标分别判断；主方案未全部达标。回撤15%～18%只是原设想，不能当作策略保证。', '',
        '## 冻结后的完整规则', '',
        '- 每周首个中国交易日收盘评估，下一交易日收盘成交；休市顺延，周中不新增信号，不把次日收盘结果表述为次日开盘。',
        '- 正常底仓60%，波段目标档位60%／75%／95%。H20269全收益收盘相对MA200乖离≤−5%时从60%升至75%，≤−10%时从75%升至95%；每周最多一档，条件不满足则延续。',
        '- 波段仓在乖离≥6%时从95%降到75%，≥8%时从75%降到60%；不因过热把正常底仓降到60%以下。档位是目标状态，不等于已成交批次数；部分成交另记。',
        '- 风险条件为`(P<MA250×0.96 且 MA250<20个交易日前的MA250) 或 vol60>20%`。波动采用60日收益样本标准差、252年化。风险优先，重置波段档位，总红利目标降为30%；目标约束不代表受成交限制时能当日卖足。',
        '- 恢复要求`P>MA250×1.01、MA250斜率非负、vol60<18%`同时满足，并连续两次周度评估确认；恢复至60%，恢复当周不再网格加仓。',
        '- 债券版国债目标为`95%−红利目标`，至少5%目标现金；现金版余款均为零息现金。511260上市前债券目标为零，留现金。没有把2%～3.5%固定收益写入回测，也没有假造短债历史。',
        '- 非零目标使用3个百分点漂移带；风险减仓和零目标绕过漂移带；先卖后买、自融资。95%是目标上限，价格波动、整手与费用会造成实际权重偏离。',
        '- 第一轮将RSI和20日涨幅排除，避免同时叠加指标。40日相对收益和波动率网格分别作为独立预登记对照，不能拿结果最高者事后改称主方案。',
        '', '## 两种账户口径', '',
        '**2016起代理账户：**H20269已含股息再投，持有部分按日历天扣0.60%年化复制费用情景；511260用原始收盘乘复权因子的总收益代理，基金费用已内含。分数等值份额，非可交易ETF历史。511260自2017-08-24上市前留现金，因此全期债券版本不能被称为十年国债ETF完整经历2016。',
        '**2019起ETF账户：**仅使用512890和511260共同存在后的原始价格；每笔100份整手，按场内登记日记录分红权益、除息日计应收、实际发放日进入可用现金。两ETF不再扣指数复制费。',
        '**共同费用：**每资产每方向`max(5元,成交额×0.10%)`，是佣金滑点合并折减，不等于真实券商报价。ETF每次成交额同时受前20个有效报价日平均成交额5%和当日成交额5%限制；历史不足20个有效报价日不得新增。部分或零成交等下周信号重算，缺报价则延后，不生成虚假成交。',
        '持有基准按起始日收盘归一化，不额外扣买卖成本；95%持有对照使用同一主动账户、时序和费用。指数、价格代理与整手账户不能混成统一成交模型的胜负结论。', '',
        '## 2016起分层对照', '',
        *table([('红利低波全收益持有',result['benchmarks']['proxy'])]+[(LABELS[s],rows[s+'__proxy']) for s in baseids]), '',
        '![全期净值与回撤](proxy_comparison.png)', '',
        '## 2019起真实ETF整手对照', '',
        *table([('512890持有价格代理',result['benchmarks']['etf'])]+[(LABELS[s],rows[s+'__etf']) for s in baseids]), '',
        '![上市后净值与回撤](etf_comparison.png)', '',
        '## 贡献与机制诊断', '',
        f"相对固定60%红利35%国债，不加风控的逆向网格使全期年化从{rows['static60_bond__proxy']['cagr']:.2%}变为{rows['grid_bond__proxy']['cagr']:.2%}，但回撤从{rows['static60_bond__proxy']['max_drawdown']:.2%}变为{rows['grid_bond__proxy']['max_drawdown']:.2%}，最差年从{rows['static60_bond__proxy']['worst_full_year']:.2%}变为{rows['grid_bond__proxy']['worst_full_year']:.2%}。收益与风险必须一起看。",
        f"主方案共有{main['grid_add_count']}次网格加档、{main['grid_reduce_count']}次网格减档，{main['risk_signal_weeks']}次周度信号处于风险状态；平均红利{main['average_red_weight']:.2%}、国债{main['average_bond_weight']:.2%}、现金{main['average_cash_weight']:.2%}。风控与深跌加仓存在重叠：跌到买线附近时可能先触发风险上限，因此本参数下网格交易机会有限。本轮保留登记主方案，不根据已见收益放宽风控。",
        f"ETF主方案共{live['trade_count']}笔成交，{live['clipped_order_count']}次订单受整手、现金或容量等约束而缩减。解除容量限制的乐观对照年化{rows['grid_risk_bond__etf__no_capacity']['cagr']:.2%}、回撤{rows['grid_risk_bond__etf__no_capacity']['max_drawdown']:.2%}；不能把这个乐观版本当成已验证实际成交结果。",
        '每年红利资产盈亏、国债资产盈亏和费用分别复算，逐日相加与账户净值变化一致，见`annual_pnl_attribution.csv`。国债贡献需要与网格增量区分，不能把债券收益当作择时超额。', '',
        '## 主方案执行压力', '',
        *table([('全期基础',main)]+[(x['id'].split('__')[-1],x) for x in result['rows']
             if x['candidate']=='grid_risk_bond' and x['mode']=='proxy' and x['id'].count('__')==2],False), '',
        *table([('ETF基础',live)]+[(x['id'].split('__')[-1],x) for x in result['rows']
             if x['candidate']=='grid_risk_bond' and x['mode']=='etf' and x['id'].count('__')==2],False), '',
        'cost2为单边成本翻倍；delay2为T+2收盘；week_end为每周末交易日收盘信号；fee1pct为指数复制费用1%；carry_gap为国债缺报价日延用前价；no_capacity为解除ETF容量限制。', '',
        '## 参数邻域与独立信号对照', '',
        *table([(LABELS[s],rows[s+'__proxy']) for s in sensitivities],False), '',
        '底仓50%版本的档位为50%／70%／95%，风险目标25%；MA180/250只改变网格均线，风险均线仍为250日。风险波动阈值按进入／退出成对列示：18%／16%或22%／20%，基准为20%／18%。波动率网格将买卖乖离阈值乘以clip(vol60/15%,0.75,1.5)。宽窄网格、波动阈值、恢复周数分别只改一个参数方向。40日相对收益对照使用H20269减H00300全收益，采用−1%／−3%加仓、5%／6%减仓；它是新增沪深300对照，未冒称Wind全A原文复现。', '',
        '## 主方案年度收益', '',
        '| 年份 | 代理主方案 | 红利全收益 | 完整年 |', '| --- | ---: | ---: | --- |']
    benchannual={a['year']:a for a in result['benchmarks']['proxy']['annual']}
    for a in main['annual']:
        lines.append(f"| {a['year']} | {a['return']:.2%} | {benchannual[a['year']]['return']:.2%} | {'是' if a['complete'] else '否，部分年'} |")
    lines.extend(['', '## 数据核查与局限', '',
        '- 511260在2018-08-17缺交易价格，腾讯原始日线也没有该日记录，基金净值表存在当日102.057。该日只用净值估值且禁止成交，不声称已证实停牌。净值公告时间未单独证明，因此保留前价估值压力；不将估值价用于交易或信号。',
        '- 新增2026年9月基金原始分红公告，核对场内登记9月17日、除息9月18日、投资者可用现金9月23日，每份1.2747元；未采用场外9月21日到账。已有四笔分红沿用逐页核验原件，五笔金额/除息日与新浪累计分红核对。',
        '- 512890发行人原件证明2021-10-22拆分登记、10月25日恢复交易，1:2拆分；新浪因子在10月25日才出现，研究按实际拆分日处理。拆分日仅估值不成交。',
        '- 补充H00300的2016年起官方历史以满足40日相对收益预热，保留完整原始响应；首次不足预热的部分回放已留档，不修改阈值。',
        '- 参数由本次讨论和已查看旧历史形成；所有新候选在完整运行前登记，但不是未见历史盲测。分段和参数邻域不能替代真实前瞻验证。未用本轮最佳收益回写主方案。',
        '- 整手、分红、容量和部分价格限制仅是模拟约束；未验证券商实际收盘成交、所有市场规则或全市场历史存续。0.60%复制费用是统一情景，不是已审计历史费率表。',
        '- 尚无合格短债/货币历史，本轮不提供该分支的实证结论。没有新增券商连接、调度、部署或生产策略切换。', '',
        '## 验证证据与复跑', '',
        f"18项针对性测试通过；57条最终回放完成，独立现金/份额/应收重建最大误差{verification['independent_ledger_max_error']:.3g}元，独立指标最大误差{verification['independent_metric_max_error']:.3g}，指标逐窗重算最大误差{verification['independent_indicator_max_error']:.3g}。另对主方案两个账户执行截断信号与账户前缀检查。逐日盈亏贡献核对通过。",
        '正式数据库、策略注册、生产指针、投资偏好及旧研究完成记录哈希保持不变。没有运行无关前后端、模型训练或全量项目测试。',
        f"冻结run：`{directory.relative_to(ROOT).as_posix()}`。数据：`{reg['data_directory']}`。完整57行指标、年度收益、分段、逐日净值、交易、订单与公司行动日志均在run目录。", '',
        '```powershell',
        '.venv/Scripts/python.exe -X utf8 -m unittest research.dividend_grid.test_strategy -v',
        '.venv/Scripts/python.exe -X utf8 -m research.dividend_grid.run --register output/dividend_grid/data_20261111_v2',
        '.venv/Scripts/python.exe -X utf8 -m research.dividend_grid.run --run <新登记目录>',
        '```',''])
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['annual','execution','personal_goals']}
                  for r in result['rows']]).to_csv(output/'all_results.csv',index=False,encoding='utf-8-sig')
    write_json(output/'summary.json',{'primary':main,'etf_primary':live,'benchmarks':result['benchmarks'],
        'base_candidates_dominating_passive':winners,'investment_admission':False,
        'future_unseen_holdout':False,'production_changed':False,'source_run':str(directory.relative_to(ROOT))})
    print(json.dumps({'report':str(output/'report.md'),'plots':[str(p) for p in plots],
                     'dominance':winners,'main_cagr':main['cagr'],'etf_cagr':live['cagr']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();build(a.run,a.output)
