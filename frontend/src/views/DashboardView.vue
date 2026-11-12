<template>
  <DevelopmentFrame :title="pageTitle" :subtitle="pageSubtitle" :snapshot="snapshot" :loading="loading" :error="error" @reload="load">
    <div class="dev-notice warning">正式服务已登记；投资准入仍按目标与审计分别核实。历史选择、完整存续标的审计与真实前瞻验证的限制继续保留。</div>
    <article v-for="strategy in snapshot.strategies" :key="strategy.id" class="dev-panel overview-strategy" :style="{ '--accent': strategy.color }">
      <div class="dev-panel-head"><div class="overview-title"><i></i><div><h2>{{ strategy.name }}</h2><p>{{ strategy.description }}</p></div></div><span class="dev-badge" :class="strategy.investment_admission ? 'pass' : 'fail'">v{{ strategy.version }} · {{ strategy.investment_admission ? '投资准入通过' : '投资准入未通过' }}</span></div>
      <div class="dev-metric-grid">
        <div class="dev-metric"><span>最差完整年度</span><strong :class="{ 'dev-negative': strategy.metrics.worst_complete_year?.return < 0 }">{{ percent(strategy.metrics.worst_complete_year?.return) }}</strong><em>{{ strategy.metrics.worst_complete_year?.year || '—' }}年 · 目标 ≥−5% <b :class="strategy.goals.worst_complete_year ? 'dev-positive' : 'dev-negative'">{{ goalText(strategy.goals.worst_complete_year) }}</b></em></div>
        <div class="dev-metric"><span>全期最大回撤</span><strong>{{ percent(strategy.metrics.max_drawdown) }}</strong><em>目标 ≤40% · {{ goalText(strategy.goals.max_drawdown) }}</em></div>
        <div class="dev-metric"><span>复合年化收益</span><strong>{{ percent(strategy.metrics.cagr) }}</strong><em>目标 ≥13.50% · {{ goalText(strategy.goals.cagr) }}</em></div>
        <div class="dev-metric"><span>{{ isIndexStrategy(strategy) ? '指数参考资金' : '当前模型资金' }}</span><strong>{{ number(strategy.metrics.final_account / 10000) }}<small> 万元</small></strong><em>起始 {{ number(strategy.initial_capital / 10000, 0) }}万元 · 累计 {{ percent(strategy.metrics.cumulative_return) }}</em></div>
      </div>
      <div v-if="isValuationStrategy(strategy)" class="overview-valuation">
        <div><span>信号估值口径</span><b>{{ valuationMetricLabel(strategy.valuation?.metric) }}</b></div>
        <div><span>信号值 / 估值日</span><b>{{ number(strategy.valuation?.value, 4) }} · {{ strategy.valuation?.trade_date || '—' }}</b></div>
        <div><span>买入上沿 / 深入线</span><b>{{ number(strategy.valuation?.entry_upper, 4) }} / {{ number(strategy.valuation?.entry_low, 4) }}</b></div>
        <div><span>清仓线</span><b>{{ number(strategy.valuation?.exit_level, 4) }} · {{ percent(strategy.valuation?.exit_quantile) }}分位</b></div>
      </div>
      <div v-if="isDividendBandStrategy(strategy)" class="overview-valuation">
        <div><span>隐含股息率 / 信号日</span><b>{{ percent(strategy.dividend_signal?.implied_yield) }} · {{ strategy.dividend_signal?.date || '—' }}</b></div>
        <div><span>红利 / 国债目标</span><b>{{ percent(strategy.dividend_signal?.red_target) }} / {{ percent(strategy.dividend_signal?.bond_target) }}</b></div>
        <div><span>现金目标</span><b>{{ percent(strategy.dividend_signal?.cash_target) }}</b></div>
        <div><span>指数口径</span><b>价格 {{ strategy.dividend_signal?.price_index || 'H30269' }} · 全收益 {{ strategy.dividend_signal?.return_index || 'H20269' }}</b></div>
      </div>
      <div v-if="!isIndexStrategy(strategy)" class="overview-account"><div><span>当前账户</span><b>{{ holdingCountText(strategy) }}</b><span>现金 {{ number(strategy.cash / 10000) }}万元</span><span v-if="strategy.dividend_receivable">分红应收 {{ number(strategy.dividend_receivable) }}元</span></div><div><span>最新信号</span><b>{{ strategy.advice.signal_date }}</b><span>{{ strategy.advice.orders_count }}笔{{ isDividendBandStrategy(strategy) ? '模型调仓' : '调仓' }} · {{ strategy.runtime.advice_label }}</span></div></div>
      <p class="dev-note">{{ strategy.account_basis }}。{{ isValuationStrategy(strategy) ? valuationCaveat(strategy) : strategy.limitations?.join('；') }}。</p>
      <div class="overview-links"><router-link :to="{ name: 'picks', query: { strategy: strategy.id } }">{{ isDividendBandStrategy(strategy) ? '查看历史七档信号 →' : '查看候选与门槛 →' }}</router-link><router-link :to="{ name: 'decision', query: { strategy: strategy.id } }" v-if="!isIndexStrategy(strategy)">核对周度调仓 →</router-link><router-link :to="{ name: 'backtest', query: { strategy: strategy.id } }">{{ isIndexStrategy(strategy) ? '指数资金曲线 →' : isDividendBandStrategy(strategy) ? '资金曲线与等值模拟 →' : '资金曲线与成交 →' }}</router-link></div>
    </article>
<section class="dev-panel">
      <div class="dev-panel-head"><div><h2>数据覆盖与审计</h2><p>用于判断结果的可复核范围；31只储存标的不等于每条策略的可交易候选。</p></div><router-link :to="{ name: 'pipeline' }">更新正式数据 →</router-link></div>
      <dl class="audit-grid"><div><dt>行情范围</dt><dd>{{ audit.start || '—' }} — {{ audit.end || snapshot.data_end }}</dd></div><div><dt>价格记录 / 储存标的</dt><dd>{{ number(audit.bars, 0) }}条 / {{ number(audit.symbols, 0) }}只</dd></div><div><dt>行情来源</dt><dd>{{ audit.sources?.join('、') || '—' }}</dd></div><div><dt>规模信息最新日期</dt><dd>{{ audit.nav_latest_information_date || '—' }}</dd></div><div><dt>规模可用原则</dt><dd>报告公告日严格早于信号日</dd></div><div><dt>数据未及时更新的标的</dt><dd>{{ audit.stale_symbols?.join('、') || '无' }}</dd></div><div><dt>历史候选完整性</dt><dd>当前整理候选列表，存续与退市完整性尚未审计</dd></div><div><dt>复权及公司行动</dt><dd>复权因子与已登记分红参与回放；完整公司行动审计尚未通过</dd></div></dl>
    </section>
  </DevelopmentFrame>
</template>
<script setup>
import { computed } from 'vue'
import DevelopmentFrame from '../components/DevelopmentFrame.vue'
import { useFormalSnapshot, number, percent } from '../composables/useFormalSnapshot'
import { isDividendBandStrategy, isIndexStrategy, isValuationStrategy, valuationCaveat, valuationMetricLabel } from '../strategyPresentation'
const { snapshot, loading, error, load } = useFormalSnapshot()
const strategyCount = computed(() => snapshot.value?.strategies?.length || 0)
const pageTitle = computed(() => strategyCount.value ? `${strategyCount.value}条正式策略研究总览` : '正式策略研究总览')
const pageSubtitle = computed(() => `当前按稳健、平衡、积极顺序展示 ${strategyCount.value || '三'} 项；积极A按周首信号、T+1收盘进行指数等值回放。先看年度亏损与回撤，再看收益；策略、账户与数据均对应本页运行快照。`)
const audit = computed(() => snapshot.value?.data_audit || {})
const goalText = value => value ? '达标' : '未达标'
const holdingCountText = strategy => isDividendBandStrategy(strategy)
  ? `${strategy.holdings?.length || 0}项指数代理`
  : `${strategy.holdings?.length || 0}只ETF`
</script>
<style scoped>
.overview-title { display: flex; align-items: flex-start; gap: 12px; }.overview-title > i { width: 3px; height: 36px; margin-top: 3px; border-radius: 3px; background: var(--accent); flex-shrink: 0; }.overview-valuation { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1px; margin-top: 20px; border: 1px solid #e7ecf0; background: #e7ecf0; border-radius: 5px; overflow: hidden; }.overview-valuation > div { display: flex; flex-direction: column; gap: 7px; padding: 12px 14px; background: #fbfcfd; min-width: 0; }.overview-valuation span { color: var(--muted); font-size: 10px; }.overview-valuation b { color: #496078; font-size: 11px; font-weight: 500; overflow-wrap: anywhere; }.overview-account { padding: 16px 0; margin-top: 22px; border-top: 1px solid var(--line); display: flex; justify-content: space-between; flex-wrap: wrap; gap: 14px; font-size: 11px; color: var(--muted); }.overview-account > div { display: flex; gap: 15px; flex-wrap: wrap; }.overview-account b { color: #425971; font-weight: 500; }.overview-links { display: flex; gap: 25px; flex-wrap: wrap; font-size: 11px; margin-top: 18px; }.audit-grid { display: grid; grid-template-columns: 1fr 1fr; margin: 0; gap: 0 30px; }.audit-grid > div { border-bottom: 1px solid #edf1f4; padding: 16px 0; }.audit-grid dt { color: var(--muted); font-size: 11px; }.audit-grid dd { color: #485d76; font-size: 12px; margin: 9px 0 0; line-height: 1.8; overflow-wrap: anywhere; }
@media(max-width: 700px) { .audit-grid { grid-template-columns: 1fr; }.overview-links { gap: 14px; }.overview-account { gap: 15px; }.overview-valuation { grid-template-columns: 1fr 1fr; } }
</style>
