<template>
  <DevelopmentFrame title="回测分析" subtitle="2016-09-12起，各100万元模型账户，统一周频与次交易日开盘执行。每日收盘估值、年度风险与已成交记录按正式运行展示。" :snapshot="snapshot" :loading="loading" :error="error" @reload="load">
    <section class="dev-panel"><div class="dev-panel-head"><div><h2>双策略 · 同起点比较</h2><p>两者账户估值方式不同，分别注明；历史结果用于研究复核。</p></div></div><div class="dev-table-scroll"><table class="dev-table"><thead><tr><th>策略 / 账户口径</th><th class="numeric">最差完整年</th><th class="numeric">最大回撤</th><th class="numeric">年化收益</th><th class="numeric">当前资金（万元）</th><th>投资准入</th></tr></thead><tbody><tr v-for="item in snapshot.strategies" :key="item.id"><td class="wrap"><b :style="{ color: item.color }">{{ item.name }}</b><small>{{ item.account_basis }}</small></td><td class="numeric dev-negative">{{ percent(item.metrics.worst_complete_year?.return) }}<small>{{ item.metrics.worst_complete_year?.year }}年</small></td><td class="numeric">{{ percent(item.metrics.max_drawdown) }}</td><td class="numeric">{{ percent(item.metrics.cagr) }}</td><td class="numeric">{{ number(item.metrics.final_account / 10000) }}</td><td><span class="dev-badge" :class="item.investment_admission ? 'pass' : 'fail'">{{ item.investment_admission ? '通过' : '未通过' }}</span></td></tr></tbody></table></div></section>
    <section class="dev-panel"><DevStrategySelector v-model="strategyId" :strategies="snapshot.strategies" /></section>
    <div v-if="detailError" class="dev-notice error" role="alert">{{ detailError }}</div>
    <div v-if="detailLoading" class="dev-loading" role="status"><span class="dev-spinner"></span>正在读取该策略完整回放…</div>
    <template v-else-if="detail">
      <section class="dev-panel backtest-chart" :style="{ '--accent': detail.strategy.color }"><div class="dev-panel-head"><div><h2>{{ detail.strategy.name }}</h2><p>{{ detail.strategy.data_start }} — {{ detail.strategy.data_end }} · {{ number(detail.strategy.metrics.observations, 0) }}个交易日 · {{ detail.strategy.account_basis }}</p></div></div><StrategyChart :key="`${detail.run_id}-${strategyId}`" :strategy="detail.strategy" /><WeeklyAllocation v-if="strategyId === 'trend200_vol15'" :strategy="detail.strategy" /></section>
      <section class="dev-panel"><div class="dev-panel-head"><div><h2>年度收益与回撤</h2><p>完整自然年用于最差年度目标；首尾不足全年单独标记。</p></div></div><div class="dev-table-scroll"><table class="dev-table"><thead><tr><th>年度</th><th>覆盖</th><th class="numeric">年度收益</th><th class="numeric">年度最大回撤</th><th>年度−5%目标</th></tr></thead><tbody><tr v-for="annual in detail.strategy.metrics.annual" :key="annual.year"><td>{{ annual.year }}</td><td>{{ annual.complete ? '完整年度' : '部分年度' }}</td><td class="numeric" :class="{ 'dev-negative': annual.return < 0 }">{{ percent(annual.return) }}</td><td class="numeric">{{ percent(annual.max_drawdown) }}</td><td><span v-if="annual.complete" class="dev-badge" :class="annual.return >= -.05 ? 'pass' : 'fail'">{{ annual.return >= -.05 ? '达标' : '未达标' }}</span><span v-else class="dev-note">单独披露</span></td></tr></tbody></table></div></section>
      <section class="dev-panel" :style="{ '--accent': detail.strategy.color }"><WeeklyHoldings :strategy="detail.strategy" /></section>
      <section class="dev-panel"><div class="dev-panel-head"><div><h2>执行规则与回放核对</h2><p>{{ detail.strategy.execution_revision }}</p></div><span class="dev-badge" :class="detail.validation?.weekly_t_plus_1 ? 'pass' : 'fail'">{{ detail.validation?.weekly_t_plus_1 ? '周频 T+1 核对通过' : '周频 T+1 待核对' }}</span></div>
        <dl class="rule-grid"><div><dt>信号与执行</dt><dd>周首交易日收盘信号，下一交易日开盘执行，触发才调仓</dd></div><div><dt>配置与趋势</dt><dd>{{ detail.strategy.description }}</dd></div><div><dt>波动与仓位</dt><dd>过去{{ policy.covariance_days || 60 }}日协方差，{{ percent(policy.vol_target) }}年化波动预算；最高总仓位 {{ percent(policy.max_gross_weight) }}</dd></div><div><dt>费用与容量</dt><dd>佣金 {{ percent(policy.commission) }}，滑点 {{ percent(policy.slippage) }}，另计冲击；成交参与率上限 {{ percent(policy.max_adv_participation) }}</dd></div><div><dt>持仓与调仓</dt><dd>{{ strategyId === 'lightgbm' ? `Top${policy.topk || 6}权益ETF；最少持有${policy.hold_days || 20}交易日、排名缓冲${policy.rank_buffer ?? 2}` : '通信不合格才允许沪深300替代；跌破趋势门槛持现金' }}；漂移阈值 {{ percent(policy.rebalance_band) }}</dd></div><div><dt>成交记录</dt><dd>{{ number(detail.validation?.trade_count ?? detail.trades.length, 0) }}笔；{{ detail.strategy.account_basis }}</dd></div></dl>
        <div v-if="issueEntries.length" class="validation-issues"><span v-for="[key, count] in issueEntries" :key="key">{{ issueName(key) }} <b>{{ number(count, 0) }}</b></span></div><p class="dev-note">行情、涨跌停、现金和容量约束可能使实际模拟成交偏离目标。15%波动预算不保证最大回撤或年度亏损。</p>
      </section>
      <section class="dev-panel"><div class="dev-panel-head"><div><h2>历史成交明细</h2><p>成交日和对应周首信号分别记录，未成交建议不计入。</p></div><span>{{ filteredTrades.length }}笔符合筛选</span></div><div class="dev-toolbar trade-controls"><label>年度<select v-model="tradeYear"><option value="">全部年度</option><option v-for="year in tradeYears" :key="year" :value="year">{{ year }}</option></select></label><label>方向<select v-model="tradeSide"><option value="">全部方向</option><option value="buy">买入</option><option value="sell">卖出</option></select></label></div><div class="dev-table-scroll"><table class="dev-table"><thead><tr><th>成交日 / 信号日</th><th>ETF</th><th>方向</th><th class="numeric">份额</th><th class="numeric">成交价</th><th class="numeric">佣金（元）</th><th class="numeric">滑点与冲击（元）</th></tr></thead><tbody><tr v-for="(trade, index) in shownTrades" :key="`${trade.date}-${trade.symbol}-${index}`"><td>{{ trade.date }}<small>信号 {{ trade.signal_date }}</small></td><td>{{ trade.name || trade.symbol }}<small v-if="trade.code">{{ trade.code }}</small></td><td>{{ sideName(trade.side) }}</td><td class="numeric">{{ number(trade.quantity, trade.quantity % 1 ? 2 : 0) }}</td><td class="numeric">{{ number(trade.price, 4) }}</td><td class="numeric">{{ number(trade.commission) }}</td><td class="numeric">{{ number(trade.slippage_cost) }}</td></tr><tr v-if="!shownTrades.length"><td colspan="7" class="dev-empty">筛选范围内没有成交记录。</td></tr></tbody></table></div><div v-if="filteredTrades.length" class="trade-pagination"><span>第{{ page }} / {{ pageCount }}页 · 每页20笔</span><div><button class="dev-button" :disabled="page === 1" @click="page--">上一页</button><button class="dev-button" :disabled="page >= pageCount" @click="page++">下一页</button></div></div></section>
    </template>
  </DevelopmentFrame>
</template>
<script setup>
import { computed, ref, watch, onBeforeUnmount } from 'vue'
import { useRoute } from 'vue-router'
import api from '../api'
import DevelopmentFrame from '../components/DevelopmentFrame.vue'
import DevStrategySelector from '../components/DevStrategySelector.vue'
import StrategyChart from '../components/StrategyChart.vue'
import WeeklyHoldings from '../components/WeeklyHoldings.vue'
import WeeklyAllocation from '../components/WeeklyAllocation.vue'
import { useFormalSnapshot, requestError, number, percent, sideName } from '../composables/useFormalSnapshot'
const { snapshot, loading, error, load } = useFormalSnapshot()
const route = useRoute(), strategyId = ref(route.query.strategy === 'trend200_vol15' ? 'trend200_vol15' : 'lightgbm')
const detail = ref(null), detailLoading = ref(false), detailError = ref(''), tradeYear = ref(''), tradeSide = ref(''), page = ref(1)
const policy = computed(() => detail.value?.definition?.policy || {})
const tradeYears = computed(() => [...new Set((detail.value?.trades || []).map(row => row.date.slice(0, 4)))].sort().reverse())
const filteredTrades = computed(() => (detail.value?.trades || []).filter(row => (!tradeYear.value || row.date.startsWith(tradeYear.value)) && (!tradeSide.value || row.side === tradeSide.value)).slice().sort((a, b) => b.date.localeCompare(a.date)))
const pageCount = computed(() => Math.max(1, Math.ceil(filteredTrades.value.length / 20)))
const shownTrades = computed(() => filteredTrades.value.slice((page.value - 1) * 20, page.value * 20))
const issueEntries = computed(() => Object.entries(detail.value?.validation?.issues || {}))
const issueName = key => ({ suspended_or_missing: '停牌 / 数据缺失', price_limit: '涨跌停限制', capacity_clipped: '容量约束', cash_clipped: '现金约束', risk_clipped: '风险约束', corporate_action_order_cancelled: '公司行动取消委托' })[key] || key
let controller, revision = 0, disposed = false
async function fetchDetail() {
  if (!snapshot.value) return
  const current = ++revision
  controller?.abort(); controller = new AbortController(); detail.value = null; detailError.value = ''; detailLoading.value = true
  try {
    const result = await api.get('/development/backtest', { params: { strategy_id: strategyId.value, run_id: snapshot.value.run_id }, signal: controller.signal, silentErrors: true })
    if (!disposed && current === revision) { detail.value = result; tradeYear.value = tradeYears.value[0] || ''; tradeSide.value = ''; page.value = 1 }
  } catch (err) { if (!disposed && current === revision && err.code !== 'ERR_CANCELED') detailError.value = requestError(err, '正式回测明细无法读取，请刷新快照。') }
  finally { if (!disposed && current === revision) detailLoading.value = false }
}
watch([strategyId, () => snapshot.value?.run_id], fetchDetail)
watch([tradeYear, tradeSide], () => { page.value = 1 })
onBeforeUnmount(() => { disposed = true; revision++; controller?.abort() })
</script>
<style scoped>
.backtest-chart :deep(.chart-frame) { margin-top: 0; padding-top: 20px; }.dev-panel :deep(.weekly-history) { margin-top: 0; padding-top: 0; border-top: 0; }.rule-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0 28px; margin: 0; }.rule-grid > div { border-bottom: 1px solid #edf1f4; padding: 15px 0; }.rule-grid dt { color: var(--muted); font-size: 11px; }.rule-grid dd { margin: 9px 0 0; font-size: 12px; color: #4a5e77; line-height: 1.9; }.validation-issues { display: flex; flex-wrap: wrap; gap: 13px 22px; margin-top: 20px; font-size: 11px; color: var(--muted); }.validation-issues b { font-weight: 500; color: #4a5e77; margin-left: 7px; }.trade-controls { margin-bottom: 20px; }.trade-pagination { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-top: 20px; color: var(--muted); font-size: 11px; }.trade-pagination > div { display: flex; gap: 9px; }
@media(max-width: 650px) { .rule-grid { grid-template-columns: 1fr; }.trade-pagination { flex-wrap: wrap; } }
</style>
