<template>
  <DevelopmentFrame title="选ETF清单" subtitle="查看估值带、趋势门槛或隐含股息率七档配置。不同策略使用各自规则，不互相套用判断口径。" :snapshot="snapshot" :loading="loading" :error="error" @reload="load">
    <section class="dev-panel"><DevStrategySelector v-model="strategyId" :strategies="snapshot.strategies" /><form class="dev-toolbar picks-controls" @submit.prevent="query()"><label>周首信号日<select v-model="date"><option value="">最新信号</option><option v-for="day in dates" :key="day" :value="day">{{ day }}</option></select></label><button class="dev-button primary" :disabled="querying">{{ querying ? '查询中…' : '查询清单' }}</button><button type="button" class="dev-button" :disabled="querying" @click="latest">最新信号</button></form></section>
    <div v-if="queryError" class="dev-notice error" role="alert">{{ queryError }}</div>
    <div v-if="querying" class="dev-loading" role="status"><span class="dev-spinner"></span>正在读取该周正式候选…</div>
    <template v-else-if="result">
      <div v-if="result.fallback" class="dev-notice warning">请求 {{ result.requested_date }}，已回退至可用周首信号 {{ result.date }}。</div>
      <section v-if="result.kind === 'index_benchmark'" class="dev-panel"><div class="dev-panel-head"><div><h2>积极策略 · 红利低波全收益</h2><p>{{ result.description }}</p></div></div><p>H20269 · {{ result.benchmark?.data_date }} · 全收益指数点位 {{ number(result.benchmark?.close) }}</p><p class="dev-note">指数包含股息再投，未扣基金费与复制成本；不提供ETF候选或调仓指令。</p></section>
      <section v-else class="dev-panel">
        <div class="dev-panel-head"><div><h2>{{ resultTitle }}</h2><p>{{ result.description }}</p></div><span>信号 {{ result.date }}</span></div>
        <DividendSignalPanel v-if="isDividendBandResult(result.kind)" :strategy="resultStrategy" :signal="result.dividend_signal || {}" :decision-detail="resultDecisionDetail" />
        <div v-if="isDividendBandResult(result.kind)" class="dev-table-scroll"><table class="dev-table"><thead><tr><th>标的 / 代码</th><th>动作</th><th>等值口径</th><th class="numeric">当前 → 目标</th><th class="numeric">模拟金额（元）</th><th>原因</th></tr></thead><tbody>
          <tr v-for="row in result.rows" :key="row.asset || row.symbol"><td>{{ row.name || assetName(row.asset) }}<small>{{ row.code || row.symbol }}</small></td><td><span class="order-side" :class="row.side">{{ sideName(row.side) }}</span></td><td>等值模拟</td><td class="numeric">{{ percent(row.current_weight) }} → {{ percent(row.target_weight) }}</td><td class="numeric">{{ number(row.estimated_amount, 0) }}</td><td class="reason">{{ row.reason }}</td></tr>
          <tr v-if="!result.rows.length"><td colspan="6" class="dev-empty">本周没有指数等值调仓动作，保持当前红利、国债与现金配置。</td></tr>
        </tbody></table></div>
        <div v-else class="dev-table-scroll"><table class="dev-table"><thead><tr><th>ETF / 代码</th><th>市场资格</th><th class="numeric">收盘价</th><template v-if="result.kind === 'trend'"><th class="numeric">200日均线</th><th class="numeric">相对均线</th><th>趋势门槛</th></template><template v-if="isValuationResult(result.kind)"><th>估值口径 / 信号值</th><th>估值买入线（上沿 / 中线 / 深入）</th><th>估值清仓线</th></template><th>判断原因</th></tr></thead><tbody>
          <tr v-for="row in result.rows" :key="row.symbol"><td>{{ row.name }}<small>{{ row.code }}</small></td><td><span class="dev-badge" :class="row.eligible ? 'pass' : 'fail'">{{ row.eligible ? '合格' : '不合格' }}</span></td><td class="numeric">{{ number(row.close, 3) }}</td><template v-if="result.kind === 'trend'"><td class="numeric">{{ number(row.sma200, 4) }}</td><td class="numeric" :class="{ 'dev-negative': row.deviation < 0 }">{{ percent(row.deviation) }}</td><td><span class="dev-badge" :class="row.trend_pass ? 'pass' : 'fail'">{{ row.trend_pass == null ? '数据不足' : row.trend_pass ? '通过' : '未通过' }}</span></td></template><template v-if="isValuationResult(result.kind)"><td class="valuation-lines"><b>{{ valuationMetricLabel(row.valuation?.metric) }}</b><span>信号值 {{ number(row.valuation?.value, 4) }} · {{ row.valuation?.trade_date || '—' }}</span></td><td class="valuation-lines"><span>上沿 {{ number(row.valuation?.entry_upper, 4) }}</span><span>中线 {{ number(row.valuation?.entry_mid, 4) }}</span><span>深入 {{ number(row.valuation?.entry_low, 4) }}</span></td><td class="valuation-lines"><b>{{ number(row.valuation?.exit_level, 4) }}</b><span>{{ percent(row.valuation?.exit_quantile) }}分位 · 指数 {{ row.valuation?.index_id || '—' }}</span></td></template><td class="reason">{{ row.reason }}</td></tr>
          <tr v-if="!result.rows.length"><td :colspan="resultColumns" class="dev-empty">本周没有可显示的候选，策略保持现金或已有账本持仓。</td></tr>
        </tbody></table></div>
        <p class="dev-note">{{ resultNote }}</p>
        <div v-if="blockedValuations.length" class="dev-notice warning valuation-warning"><span v-for="row in blockedValuations" :key="row.symbol">{{ row.code }}：{{ row.valuation.blocked_reason }}</span></div>
      </section>
    </template>
    <div v-else-if="!querying && !queryError" class="dev-empty">暂无可用候选清单，请到数据管道重建正式快照。</div>
  </DevelopmentFrame>
</template>
<script setup>
import { ref, computed, watch, onBeforeUnmount } from 'vue'
import { useRoute } from 'vue-router'
import api from '../api'
import DividendSignalPanel from '../components/DividendSignalPanel.vue'
import DevelopmentFrame from '../components/DevelopmentFrame.vue'
import DevStrategySelector from '../components/DevStrategySelector.vue'
import { useFormalSnapshot, requestError, number, percent, sideName } from '../composables/useFormalSnapshot'
import { isDividendBandResult, isValuationResult, resolveStrategyId, valuationCaveat, valuationMetricLabel } from '../strategyPresentation'
const route = useRoute()
const { snapshot, loading, error, load } = useFormalSnapshot()
const strategyId = ref(typeof route.query.strategy === 'string' && route.query.strategy ? route.query.strategy : 'csi300_pe_equal')
const date = ref(''), dates = ref([]), result = ref(null), querying = ref(false), queryError = ref('')
const selectedStrategy = computed(() => snapshot.value?.strategies?.find(strategy => strategy.id === strategyId.value))
const resultStrategy = computed(() => {
  const strategy = selectedStrategy.value
  const selected = result.value
  if (!strategy || !selected || !isDividendBandResult(selected.kind)) return strategy
  return {
    ...strategy,
    holdings: selected.holdings || [],
    cash: selected.cash,
    metrics: { ...strategy.metrics, final_account: selected.account },
  }
})
const resultDecisionDetail = computed(() => {
  const signal = result.value?.dividend_signal
  if (!signal) return ''
  return `隐含股息率${percent(signal.implied_yield)}；红利${percent(signal.red_target)}、国债${percent(signal.bond_target)}、现金${percent(signal.cash_target)}`
})
const resultTitle = computed(() => isDividendBandResult(result.value?.kind)
  ? '隐含股息率七档 · 红利 / 国债配置'
  : isValuationResult(result.value?.kind)
    ? '单指数估值带 · 买入与清仓线'
    : '固定配置 · 资格与趋势')
const resultColumns = computed(() => isDividendBandResult(result.value?.kind) ? 4 : isValuationResult(result.value?.kind) ? 7 : result.value?.kind === 'trend' ? 6 : 4)
const blockedValuations = computed(() => (result.value?.rows || []).filter(row => row.valuation?.blocked_reason))
const resultNote = computed(() => {
  if (isDividendBandResult(result.value?.kind)) return '红利端以H20269全收益指数分数等值份额回放，国债端使用SH511260复权等值收益；只描述七档目标与模型金额，不换算ETF整数份额。历史基础情景与代理口径不保证优于其他配置。'
  if (isValuationResult(result.value?.kind)) return `估值线按信号日当时口径计算，用于描述分批买入与清仓边界。${valuationCaveat(selectedStrategy.value)}`
  return '均线用复权收盘价计算，再换算为当日原价等值。通信跌破均线时持现金，只有市场资格失效时才允许沪深300替代。'
})
const assetName = asset => ({ red: '红利', bond: '国债', cash: '现金' })[asset] || asset || '—'
let controller, revision = 0, disposed = false
async function query(resetDates = false) {
  if (!snapshot.value || !snapshot.value.strategies.some(s => s.id === strategyId.value)) return
  const current = ++revision, id = strategyId.value, runId = snapshot.value.run_id
  controller?.abort(); controller = new AbortController()
  const options = { signal: controller.signal, silentErrors: true }
  querying.value = true; queryError.value = ''; result.value = null
  try {
    if (resetDates) {
      const available = await api.get('/development/picks/dates', { ...options, params: { strategy_id: id, run_id: runId } })
      if (disposed || current !== revision) return
      dates.value = [...available.dates].sort().reverse()
    }
    const data = await api.get('/development/picks', { ...options, params: { strategy_id: id, date: date.value || undefined, run_id: runId } })
    if (!disposed && current === revision) result.value = data
  } catch (err) { if (!disposed && current === revision && err.code !== 'ERR_CANCELED') queryError.value = requestError(err, '候选查询失败，请刷新快照后重试。') }
  finally { if (!disposed && current === revision) querying.value = false }
}
function latest() { date.value = ''; query() }
watch([strategyId, () => snapshot.value?.run_id], () => { date.value = ''; dates.value = []; query(true) })
watch(() => snapshot.value?.strategies, strategies => {
  if (strategies?.length && !strategies.some(strategy => strategy.id === strategyId.value)) strategyId.value = resolveStrategyId('', strategies)
}, { immediate: true })
onBeforeUnmount(() => { disposed = true; revision++; controller?.abort() })
</script>
<style scoped>
.picks-controls { margin-top: 23px; border-top: 1px solid var(--line); padding-top: 20px; }
.valuation-lines { min-width: 135px; }.valuation-lines b, .valuation-lines span { display: block; }.valuation-lines b { color: #465d75; font-weight: 500; }.valuation-lines span { color: var(--muted); font-size: 10px; line-height: 1.8; }.valuation-warning { display: grid; gap: 5px; margin-top: 16px; }
.order-side { display: inline-block; padding: 5px 8px; border-radius: 4px; font-size: 11px; background: #f2f5f7; color: #718394; }.order-side.buy { background: #edf4fd; color: #2b6ad6; }.order-side.sell { background: #f8eeee; color: #ad635c; }
</style>
