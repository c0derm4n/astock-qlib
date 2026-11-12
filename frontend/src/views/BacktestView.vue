<template>
  <DevelopmentFrame title="回测分析" subtitle="2016-09-12起，各100万元起始；稳健与平衡策略按周首信号、次日开盘回放，积极策略按周首信号、下一交易日收盘进行指数等值回放。年度风险与账户口径分别展示。" :snapshot="snapshot" :loading="loading" :error="error" @reload="load">
    <section class="dev-panel"><div class="dev-panel-head"><div><h2>正式策略 · 同起点比较</h2><p>各策略账户估值方式分别注明；历史结果仅用于研究复核，等权估值策略另有历史重建限制。</p></div></div><div class="dev-table-scroll"><table class="dev-table"><thead><tr><th>策略 / 账户口径</th><th class="numeric">最差完整年</th><th class="numeric">最大回撤</th><th class="numeric">年化收益</th><th class="numeric">当前资金（万元）</th><th>投资准入</th></tr></thead><tbody><tr v-for="item in snapshot.strategies" :key="item.id"><td class="wrap"><b :style="{ color: item.color }">{{ item.name }}</b><small>{{ item.account_basis }}</small></td><td class="numeric" :class="{ 'dev-negative': item.metrics.worst_complete_year?.return < 0 }">{{ percent(item.metrics.worst_complete_year?.return) }}<small>{{ item.metrics.worst_complete_year?.year || '—' }}年</small></td><td class="numeric">{{ percent(item.metrics.max_drawdown) }}</td><td class="numeric">{{ percent(item.metrics.cagr) }}</td><td class="numeric">{{ number(item.metrics.final_account / 10000) }}</td><td><span class="dev-badge" :class="item.investment_admission ? 'pass' : 'fail'">{{ item.investment_admission ? '通过' : '未通过' }}</span></td></tr></tbody></table></div></section>
    <section class="dev-panel"><DevStrategySelector v-model="strategyId" :strategies="snapshot.strategies" /></section>
    <div v-if="detailError" class="dev-notice error" role="alert">{{ detailError }}</div>
    <div v-if="detailLoading" class="dev-loading" role="status"><span class="dev-spinner"></span>正在读取该策略完整回放…</div>
    <template v-else-if="detail">
      <section class="dev-panel backtest-chart" :style="{ '--accent': detail.strategy.color }"><div class="dev-panel-head"><div><h2>{{ detail.strategy.name }}</h2><p>{{ detail.strategy.data_start }} — {{ detail.strategy.data_end }} · {{ number(detail.strategy.metrics.observations, 0) }}个交易日 · {{ detail.strategy.account_basis }}</p></div></div><DividendSignalPanel v-if="isDividendBand" :strategy="detail.strategy" /><div v-if="isValuation" class="backtest-valuation"><div><span>信号估值口径</span><b>{{ valuationMetricLabel(detail.strategy.valuation?.metric) }}</b></div><div><span>信号值 / 估值日</span><b>{{ number(detail.strategy.valuation?.value, 4) }} · {{ detail.strategy.valuation?.trade_date || '—' }}</b></div><div><span>买入上沿 / 中线 / 深入线</span><b>{{ number(detail.strategy.valuation?.entry_upper, 4) }} / {{ number(detail.strategy.valuation?.entry_mid, 4) }} / {{ number(detail.strategy.valuation?.entry_low, 4) }}</b></div><div><span>清仓线</span><b>{{ number(detail.strategy.valuation?.exit_level, 4) }} · {{ percent(detail.strategy.valuation?.exit_quantile) }}分位</b></div></div><StrategyChart :key="`${detail.run_id}-${strategyId}`" :strategy="detail.strategy" /><WeeklyAllocation v-if="!isIndex" :strategy="detail.strategy" /></section>
      <section class="dev-panel"><div class="dev-panel-head"><div><h2>年度收益与回撤</h2><p>完整自然年用于最差年度目标；首尾不足全年单独标记。</p></div></div><div class="dev-table-scroll"><table class="dev-table"><thead><tr><th>年度</th><th>覆盖</th><th class="numeric">年度收益</th><th class="numeric">年度最大回撤</th><th>年度−5%目标</th></tr></thead><tbody><tr v-for="annual in detail.strategy.metrics.annual" :key="annual.year"><td>{{ annual.year }}</td><td>{{ annual.complete ? '完整年度' : '部分年度' }}</td><td class="numeric" :class="{ 'dev-negative': annual.return < 0 }">{{ percent(annual.return) }}</td><td class="numeric">{{ percent(annual.max_drawdown) }}</td><td><span v-if="annual.complete" class="dev-badge" :class="annual.return >= -.05 ? 'pass' : 'fail'">{{ annual.return >= -.05 ? '达标' : '未达标' }}</span><span v-else class="dev-note">单独披露</span></td></tr></tbody></table></div></section>
      <section v-if="!isIndex" class="dev-panel" :style="{ '--accent': detail.strategy.color }"><WeeklyHoldings :strategy="detail.strategy" /></section>
      <section v-if="isIndex" class="dev-panel"><div class="dev-panel-head"><div><h2>全收益指数参考</h2><p>H20269 · 100%持有，含股息再投，未扣基金费与复制成本。</p></div></div><p class="dev-note">{{ detail.strategy.account_basis }}；不生成ETF订单，周频T+1与真实成交账本不适用。</p></section>
      <section v-else class="dev-panel"><div class="dev-panel-head"><div><h2>执行规则与回放核对</h2><p>{{ detail.strategy.execution_revision }}</p></div><span class="dev-badge" :class="detail.validation?.weekly_t_plus_1 ? 'pass' : 'fail'">{{ detail.validation?.weekly_t_plus_1 ? `周频 T+1 ${isDividendBand ? '收盘' : ''}核对通过` : '周频 T+1 待核对' }}</span></div>
        <dl class="rule-grid"><div><dt>信号与执行</dt><dd>{{ executionRuleText }}</dd></div><div><dt>配置与规则</dt><dd>{{ detail.strategy.description }}</dd></div><div><dt>{{ positionRuleTitle }}</dt><dd>{{ positionRuleText }}</dd></div><div><dt>{{ isDividendBand ? '费用与复制' : '费用与容量' }}</dt><dd>{{ costRuleText }}</dd></div><div><dt>账户与调仓</dt><dd>{{ accountRuleText }}</dd></div><div><dt>记录口径</dt><dd>{{ tradeCountText }}；{{ detail.strategy.account_basis }}</dd></div></dl>
        <div v-if="issueEntries.length" class="validation-issues"><span v-for="[key, count] in issueEntries" :key="key">{{ issueName(key) }} <b>{{ number(count, 0) }}</b></span></div><p class="dev-note">{{ ruleFootnote }}</p>
      </section>
      <section v-if="!isIndex" class="dev-panel"><div class="dev-panel-head"><div><h2>{{ isDividendBand ? '历史模拟调仓明细' : '历史成交明细' }}</h2><p>{{ isDividendBand ? '按周首信号和下一交易日收盘记录指数等值调整；不代表ETF份额成交。' : '成交日和对应周首信号分别记录，未成交建议不计入。' }}</p></div><span>{{ filteredTrades.length }}笔符合筛选</span></div><div class="dev-toolbar trade-controls"><label>年度<select v-model="tradeYear"><option value="">全部年度</option><option v-for="year in tradeYears" :key="year" :value="year">{{ year }}</option></select></label><label>方向<select v-model="tradeSide"><option value="">全部方向</option><option value="buy">买入</option><option value="sell">卖出</option></select></label></div><div class="dev-table-scroll"><table class="dev-table"><thead><tr><th>{{ isDividendBand ? '回放日 / 信号日' : '成交日 / 信号日' }}</th><th>标的</th><th>方向</th><th class="numeric">{{ isDividendBand ? '等值口径' : '份额' }}</th><th class="numeric">{{ isDividendBand ? '等值口径' : '成交价' }}</th><th class="numeric">费用（元）</th><th class="numeric">滑点与冲击（元）</th></tr></thead><tbody><tr v-for="(trade, index) in shownTrades" :key="`${trade.date}-${trade.symbol}-${index}`"><td>{{ trade.date }}<small>信号 {{ trade.signal_date }}</small></td><td>{{ tradeName(trade) }}<small v-if="tradeCode(trade)">{{ tradeCode(trade) }}</small></td><td>{{ sideName(trade.side) }}</td><td class="numeric">{{ tradeQuantity(trade) }}</td><td class="numeric">{{ tradePrice(trade) }}</td><td class="numeric">{{ tradeFee(trade) }}</td><td class="numeric">{{ number(trade.slippage_cost) }}</td></tr><tr v-if="!shownTrades.length"><td colspan="7" class="dev-empty">筛选范围内没有{{ isDividendBand ? '模拟调仓' : '成交' }}记录。</td></tr></tbody></table></div><div v-if="filteredTrades.length" class="trade-pagination"><span>第{{ page }} / {{ pageCount }}页 · 每页20笔</span><div><button class="dev-button" :disabled="page === 1" @click="page--">上一页</button><button class="dev-button" :disabled="page >= pageCount" @click="page++">下一页</button></div></div></section>
    </template>
  </DevelopmentFrame>
</template>
<script setup>
import { computed, ref, watch, onBeforeUnmount } from 'vue'
import { useRoute } from 'vue-router'
import api from '../api'
import DividendSignalPanel from '../components/DividendSignalPanel.vue'
import DevelopmentFrame from '../components/DevelopmentFrame.vue'
import DevStrategySelector from '../components/DevStrategySelector.vue'
import StrategyChart from '../components/StrategyChart.vue'
import WeeklyHoldings from '../components/WeeklyHoldings.vue'
import WeeklyAllocation from '../components/WeeklyAllocation.vue'
import { useFormalSnapshot, requestError, number, percent, sideName } from '../composables/useFormalSnapshot'
import { executionBasisLabel, isDividendBandStrategy, isIndexStrategy, isValuationStrategy, presentStrategy, resolveStrategyId, valuationCaveat, valuationMetricLabel } from '../strategyPresentation'
const { snapshot, loading, error, load } = useFormalSnapshot()
const route = useRoute()
const strategyId = ref(typeof route.query.strategy === 'string' && route.query.strategy ? route.query.strategy : 'csi300_pe_equal')
const detail = ref(null), detailLoading = ref(false), detailError = ref(''), tradeYear = ref(''), tradeSide = ref(''), page = ref(1)
const policy = computed(() => detail.value?.definition?.policy || {})
const activeStrategy = computed(() => detail.value?.strategy || snapshot.value?.strategies?.find(item => item.id === strategyId.value))
const isIndex = computed(() => isIndexStrategy(activeStrategy.value))
const isValuation = computed(() => isValuationStrategy(activeStrategy.value))
const isDividendBand = computed(() => isDividendBandStrategy(activeStrategy.value))
const positionRuleTitle = computed(() => isDividendBand.value ? '七档信号与配置' : isValuation.value ? '估值带与分批' : '持仓与调仓')
const executionRuleText = computed(() => isDividendBand.value
  ? `周首交易日收盘信号；${executionBasisLabel(activeStrategy.value?.advice?.execution_basis)}，按七档目标和5个百分点漂移带判断是否调整`
  : '周首交易日收盘信号，下一交易日开盘执行，触发才调仓')
const positionRuleText = computed(() => {
  if (isDividendBand.value) {
    const signal = activeStrategy.value?.dividend_signal || {}
    return `隐含股息率 ${percent(signal.implied_yield)}；红利目标 ${percent(signal.red_target)}、国债 ${percent(signal.bond_target)}、现金 ${percent(signal.cash_target)}。价格指数 ${signal.price_index || 'H30269'}，全收益指数 ${signal.return_index || 'H20269'}。`
  }
  if (isValuation.value) {
    const valuation = activeStrategy.value?.valuation || {}
    return `信号值 ${number(valuation.value, 4)}；买入上沿 ${number(valuation.entry_upper, 4)}、中线 ${number(valuation.entry_mid, 4)}、深入线 ${number(valuation.entry_low, 4)}；清仓线 ${number(valuation.exit_level, 4)}。单ETF分批动作还需检查批次、冷却、现金与成交约束。`
  }
  return `过去${policy.value.covariance_days || 60}日协方差，${percent(policy.value.vol_target)}年化波动预算；最高总仓位 ${percent(policy.value.max_gross_weight)}；通信不合格才允许沪深300替代，跌破趋势门槛持现金。`
})
const costRuleText = computed(() => isDividendBand.value
  ? '基础交易费用0.10%单边、单笔最低5元；红利端另计0.60%/年复制费。'
  : `佣金 ${percent(policy.value.commission)}，滑点 ${percent(policy.value.slippage)}，另计冲击；成交参与率上限 ${percent(policy.value.max_adv_participation)}`)
const accountRuleText = computed(() => {
  if (isDividendBand.value) return '最高95%可配置仓位；国债目标=95%−红利目标；SH511260上市前国债端以现金替代。'
  if (isValuation.value) return `${valuationCaveat(activeStrategy.value)}；漂移或触发阈值以正式协议为准`
  return '按正式配置与趋势门槛形成目标账户；漂移或触发阈值以正式协议为准'
})
const tradeCountText = computed(() => `${number(detail.value?.validation?.trade_count ?? detail.value?.trades?.length, 0)}${isDividendBand.value ? '条模拟调整' : '笔成交'}`)
const ruleFootnote = computed(() => isDividendBand.value
  ? '历史基础情景及代理口径用于研究复核；模型金额仅作调仓参考，不保证收益、回撤或优于其他配置。'
  : isValuation.value
    ? `${valuationCaveat(activeStrategy.value)} 估值线只描述分批边界，不保证收益或回撤。`
    : '行情、涨跌停、现金和容量约束可能使实际模拟成交偏离目标。15%波动预算不保证最大回撤或年度亏损。')
const tradeYears = computed(() => [...new Set((detail.value?.trades || []).map(row => row.date.slice(0, 4)))].sort().reverse())
const filteredTrades = computed(() => (detail.value?.trades || []).filter(row => (!tradeYear.value || row.date.startsWith(tradeYear.value)) && (!tradeSide.value || row.side === tradeSide.value)).slice().sort((a, b) => b.date.localeCompare(a.date)))
const pageCount = computed(() => Math.max(1, Math.ceil(filteredTrades.value.length / 20)))
const shownTrades = computed(() => filteredTrades.value.slice((page.value - 1) * 20, page.value * 20))
const issueEntries = computed(() => Object.entries(detail.value?.validation?.issues || {}))
const issueName = key => ({ suspended_or_missing: '停牌 / 数据缺失', price_limit: '涨跌停限制', capacity_clipped: '容量约束', cash_clipped: '现金约束', risk_clipped: '风险约束', corporate_action_order_cancelled: '公司行动取消委托' })[key] || key
const tradeQuantity = trade => isDividendBand.value || trade.quantity == null
  ? '等值模拟'
  : number(trade.quantity, Number(trade.quantity) % 1 ? 2 : 0)
const tradePrice = trade => isDividendBand.value || trade.price == null
  ? '等值模拟'
  : number(trade.price, 4)
const tradeFee = trade => number(trade.cost ?? trade.commission)
const tradeName = trade => isDividendBand.value
  ? ({ red: '红利低波（指数等值）', bond: '十年国债（复权等值）' }[trade.asset] || trade.symbol)
  : trade.name || trade.symbol
const tradeCode = trade => isDividendBand.value ? trade.symbol : trade.code || trade.symbol
let controller, revision = 0, disposed = false
async function fetchDetail() {
  if (!snapshot.value || !snapshot.value.strategies.some(s => s.id === strategyId.value)) return
  const current = ++revision
  controller?.abort(); controller = new AbortController(); detail.value = null; detailError.value = ''; detailLoading.value = true
  try {
    const result = await api.get('/development/backtest', { params: { strategy_id: strategyId.value, run_id: snapshot.value.run_id }, signal: controller.signal, silentErrors: true })
    if (!disposed && current === revision) { detail.value = { ...result, strategy: presentStrategy(result.strategy) }; tradeYear.value = tradeYears.value[0] || ''; tradeSide.value = ''; page.value = 1 }
  } catch (err) { if (!disposed && current === revision && err.code !== 'ERR_CANCELED') detailError.value = requestError(err, '正式回测明细无法读取，请刷新快照。') }
  finally { if (!disposed && current === revision) detailLoading.value = false }
}
watch([strategyId, () => snapshot.value?.run_id], fetchDetail)
watch(() => snapshot.value?.strategies, strategies => {
  if (strategies?.length && !strategies.some(item => item.id === strategyId.value)) strategyId.value = resolveStrategyId('', strategies)
}, { immediate: true })
watch([tradeYear, tradeSide], () => { page.value = 1 })
onBeforeUnmount(() => { disposed = true; revision++; controller?.abort() })
</script>
<style scoped>
.backtest-valuation { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1px; margin-top: 18px; border: 1px solid #e6ebf0; background: #e6ebf0; border-radius: 5px; overflow: hidden; }.backtest-valuation > div { display: flex; flex-direction: column; gap: 7px; padding: 12px 14px; background: #fafbfd; min-width: 0; }.backtest-valuation span { color: var(--muted); font-size: 10px; }.backtest-valuation b { color: #496078; font-size: 11px; font-weight: 500; overflow-wrap: anywhere; }
.backtest-chart :deep(.chart-frame) { margin-top: 0; padding-top: 20px; }.dev-panel :deep(.weekly-history) { margin-top: 0; padding-top: 0; border-top: 0; }.rule-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0 28px; margin: 0; }.rule-grid > div { border-bottom: 1px solid #edf1f4; padding: 15px 0; }.rule-grid dt { color: var(--muted); font-size: 11px; }.rule-grid dd { margin: 9px 0 0; font-size: 12px; color: #4a5e77; line-height: 1.9; }.validation-issues { display: flex; flex-wrap: wrap; gap: 13px 22px; margin-top: 20px; font-size: 11px; color: var(--muted); }.validation-issues b { font-weight: 500; color: #4a5e77; margin-left: 7px; }.trade-controls { margin-bottom: 20px; }.trade-pagination { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-top: 20px; color: var(--muted); font-size: 11px; }.trade-pagination > div { display: flex; gap: 9px; }
@media(max-width: 760px) { .backtest-valuation { grid-template-columns: 1fr 1fr; } }
@media(max-width: 650px) { .rule-grid { grid-template-columns: 1fr; }.trade-pagination { flex-wrap: wrap; } }
</style>
