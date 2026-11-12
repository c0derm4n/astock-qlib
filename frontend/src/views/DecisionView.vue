<template>
  <DevelopmentFrame title="周度调仓" subtitle="在所选策略的正式账本或指数等值回放中核对周度建议；普通账户试算仅在策略声明支持时开放，各策略不会互相套用计算逻辑。" :snapshot="snapshot" :loading="loading" :error="error" @reload="load">
    <section class="dev-panel"><DevStrategySelector v-model="strategyId" :strategies="snapshot.strategies" /></section>
    <template v-if="strategy">
      <section v-if="isIndex" class="dev-panel"><div class="dev-panel-head"><div><h2>{{ strategy.name }} · 指数参考</h2><p>{{ strategy.account_basis }}</p></div></div><p>H20269最新点位 {{ number(strategy.benchmark?.close) }}；100%全收益指数持有参考，不生成ETF订单。</p><p class="dev-note">{{ strategy.runtime.advice_label }}。{{ previewReason }}</p><router-link :to="{ name: 'backtest', query: { strategy: strategy.id } }">查看指数历史与回撤 →</router-link></section>
      <template v-else>
      <section class="dev-panel" :style="{ '--accent': strategy.color }">
        <div class="dev-panel-head"><div><h2>{{ strategy.name }} · 最新调仓建议</h2><p>当前展示模型账户建议；不代表真实账户持仓或成交。</p></div><span class="dev-badge" :class="strategy.runtime.advice_status !== 'pending' ? 'fail' : ''">{{ strategy.runtime.advice_label }}</span></div>
        <DividendSignalPanel v-if="isDividendBand" :strategy="strategy" />
        <div class="decision-meta"><div><span>信号收盘</span><b>{{ strategy.advice.signal_date }}</b></div><div><span>{{ isDividendBand ? '模拟执行收盘' : '计划开盘' }}</span><b>{{ strategy.advice.execution_date }}</b></div><div><span>{{ isDividendBand ? '模型调仓' : '调仓指令' }}</span><b>{{ strategy.advice.orders_count }}笔</b></div><div><span>下次评估</span><b>{{ strategy.runtime.next_signal_date }} 19:00</b></div></div>
        <div v-if="isValuation" class="valuation-summary"><div><span>信号估值口径</span><b>{{ valuationMetricLabel(strategy.valuation?.metric) }}</b></div><div><span>信号值 / 估值日</span><b>{{ number(strategy.valuation?.value, 4) }} · {{ strategy.valuation?.trade_date || '—' }}</b></div><div><span>买入上沿 / 中线 / 深入线</span><b>{{ number(strategy.valuation?.entry_upper, 4) }} / {{ number(strategy.valuation?.entry_mid, 4) }} / {{ number(strategy.valuation?.entry_low, 4) }}</b></div><div><span>清仓线</span><b>{{ number(strategy.valuation?.exit_level, 4) }} · {{ percent(strategy.valuation?.exit_quantile) }}分位</b></div></div>
        <p v-if="isValuation" class="dev-note valuation-caveat">{{ valuationCaveatText }}</p>
        <div v-if="strategy.runtime.advice_status === 'stale'" class="dev-notice warning">行情尚未覆盖应有的完整交易日，当前建议时效不足。{{ strategy.runtime.advice_label }}。请先到数据管道更新行情与正式建议。</div>
        <div v-else-if="strategy.runtime.advice_status === 'elapsed'" class="dev-notice warning">本周计划{{ isDividendBand ? '收盘（15:00）' : '开盘' }}执行窗口已过，当前建议供历史核对。请在下一次周度评估后核对新信号。</div>
        <div v-if="strategy.advice.blocked" class="dev-notice warning"><b>当前建议已阻断。</b> {{ strategy.advice.blocked_reason || '缺少形成调仓指令所需的估值或账本数据。' }}</div>
        <div class="dev-table-scroll"><table class="dev-table"><thead><tr><th>标的 / 代码</th><th>动作</th><th class="numeric">{{ isDividendBand ? '等值份额' : '份额' }}</th><th class="numeric">参考价</th><th class="numeric">{{ isDividendBand ? '模拟金额（元）' : '估算金额（元）' }}</th><th class="numeric">当前 → 目标</th><th>原因</th></tr></thead><tbody><tr v-for="row in strategy.advice.rows || []" :key="row.asset || row.symbol"><td>{{ row.name }}<small>{{ row.code || row.symbol }}</small></td><td><span class="order-side" :class="row.side">{{ sideName(row.side) }}</span></td><td class="numeric">{{ quantityText(row) }}</td><td class="numeric">{{ number(row.reference_price, 3) }}</td><td class="numeric">{{ amountText(row) }}</td><td class="numeric">{{ percent(row.current_weight) }} → {{ percent(row.target_weight) }}</td><td class="reason">{{ row.reason }}</td></tr><tr v-if="!(strategy.advice.rows || []).length"><td colspan="7" class="dev-empty">{{ emptyAdviceText }}</td></tr></tbody></table></div>
        <p class="dev-note">{{ adviceNote }}</p>
        <router-link class="decision-link" :to="{ name: 'picks', query: { strategy: strategy.id } }">查看该策略候选与门槛 →</router-link>
      </section>
      <section class="dev-panel"><div class="dev-panel-head"><div><h2>当前{{ isDividendBand ? '指数代理' : '策略账本' }}持仓</h2><p>{{ strategy.data_end }} 收盘估值 · {{ strategy.account_basis }}</p></div><span>{{ holdingCountText }}</span></div><div class="dev-table-scroll"><table class="dev-table"><thead><tr><th>标的 / 代码</th><th v-if="!isDividendBand" class="numeric">持有份额</th><th v-if="!isDividendBand" class="numeric">收盘价</th><th v-else class="numeric">代理口径</th><th class="numeric">{{ isDividendBand ? '模拟金额（元）' : '市值（元）' }}</th><th class="numeric">仓位</th></tr></thead><tbody><tr v-for="holding in strategy.holdings" :key="holding.asset || holding.symbol"><td>{{ holding.name }}<small>{{ holding.code || holding.symbol }}</small></td><td v-if="!isDividendBand" class="numeric">{{ number(holding.quantity) }}</td><td v-if="!isDividendBand" class="numeric">{{ number(holding.price, 3) }}</td><td v-else class="numeric">{{ positionSize(holding) }}</td><td class="numeric">{{ number(holding.value) }}</td><td class="numeric">{{ percent(holding.weight) }}</td></tr><tr v-if="!strategy.holdings.length"><td :colspan="isDividendBand ? 4 : 5" class="dev-empty">空仓，当前{{ isDividendBand ? '指数代理组合' : '策略账本' }}未持有仓位。</td></tr><tr><td>可用现金</td><td v-if="!isDividendBand"></td><td v-if="!isDividendBand"></td><td v-else class="numeric">现金</td><td class="numeric">{{ number(strategy.cash) }}</td><td class="numeric">{{ percent(strategy.cash / strategy.metrics.final_account) }}</td></tr><tr v-if="strategy.dividend_receivable"><td>分红应收（尚不可用）</td><td v-if="!isDividendBand"></td><td v-if="!isDividendBand"></td><td v-else class="numeric">现金</td><td class="numeric">{{ number(strategy.dividend_receivable) }}</td><td class="numeric">{{ percent(strategy.dividend_receivable / strategy.metrics.final_account) }}</td></tr></tbody></table></div></section>
      <section v-if="previewEnabled" class="dev-panel"><div class="dev-panel-head"><div><h2>用我的账户试算</h2><p>输入 {{ strategy.advice.signal_date }} 收盘时的现金与实际份额。输入不保存，结果供核对，不发送券商订单。</p></div><span>绑定本页正式快照</span></div>
        <div v-if="universeError" class="dev-notice error" role="alert">{{ universeError }}</div>
        <form @submit.prevent="calculate"><label class="dev-field cash-field" for="preview-cash">可用现金（元）<input id="preview-cash" v-model.number="cash" type="number" min="0" max="1000000000000" step="0.01" required></label><div v-for="(holding, index) in holdings" :key="holding.key" class="account-input-row"><label class="dev-field">ETF代码<select v-model="holding.symbol" required :disabled="universeLoading"><option value="" disabled>{{ universeLoading ? '读取候选中…' : '选择ETF' }}</option><option v-for="symbol in universe" :key="symbol.symbol" :value="symbol.symbol">{{ symbol.code }} {{ symbol.name }}{{ symbol.eligible ? '' : '（资格不合格）' }}</option></select></label><label class="dev-field">持有份额<input v-model.number="holding.quantity" type="number" min="0" max="10000000000" step="1" required></label><label class="dev-field">已持有交易日<input v-model.number="holding.holding_days" type="number" min="0" max="10000" step="1" required></label><button type="button" class="remove-position" @click="holdings.splice(index, 1)" :aria-label="`移除第${index + 1}行持仓`">移除</button></div><div class="account-buttons"><button type="button" class="dev-button" :disabled="calculating || universeLoading || !universe.length || holdings.length >= universe.length" @click="addHolding">＋ 添加持仓</button><button class="dev-button primary" :disabled="calculating || universeLoading || !!universeError">{{ calculating ? '试算中…' : '生成账户试算' }}</button></div></form>
        <div v-if="calcError" class="dev-notice error preview-error" role="alert">{{ calcError }}</div>
        <div v-if="calculation" class="preview-result"><p class="dev-note">输入账户估值 {{ number(calculation.account) }}元 · {{ calculation.basis }}。{{ calculation.runtime?.advice_label }}</p><div class="dev-table-scroll"><table class="dev-table"><thead><tr><th>ETF / 代码</th><th>动作</th><th class="numeric">份额</th><th class="numeric">参考价</th><th class="numeric">目标权重</th><th>原因</th></tr></thead><tbody><tr v-for="row in calculation.rows" :key="row.symbol"><td>{{ row.name }}<small>{{ row.code }}</small></td><td>{{ sideName(row.side) }}</td><td class="numeric">{{ number(row.quantity, row.quantity % 1 ? 2 : 0) }}</td><td class="numeric">{{ number(row.reference_price, 3) }}</td><td class="numeric">{{ percent(row.target_weight) }}</td><td class="reason">{{ row.reason }}</td></tr><tr v-if="!calculation.rows.length"><td colspan="6" class="dev-empty">无调仓指令，保持现金或已有合格持仓。</td></tr></tbody></table></div></div>
      </section>
      <section v-else class="dev-panel preview-unavailable"><div class="dev-panel-head"><div><h2>账户试算暂不可用</h2><p>{{ previewReason }}</p></div><span>当前策略不支持</span></div><div class="unavailable-reason"><b>为什么不能试算</b><p>{{ previewUnavailableDetail }}</p></div></section>
      </template>
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
import { useFormalSnapshot, requestError, number, percent, sideName } from '../composables/useFormalSnapshot'
import {
  accountPreviewAvailable,
  accountPreviewReason,
  executionBasisLabel,
  isDividendBandStrategy,
  isValuationStrategy,
  isIndexStrategy,
  resolveStrategyId,
  valuationCaveat,
  valuationMetricLabel,
} from '../strategyPresentation'
const { snapshot, loading, error, load } = useFormalSnapshot()
const route = useRoute()
const strategyId = ref(typeof route.query.strategy === 'string' && route.query.strategy ? route.query.strategy : 'csi300_pe_equal')
const strategy = computed(() => snapshot.value?.strategies.find(item => item.id === strategyId.value))
const isIndex = computed(() => isIndexStrategy(strategy.value))
const isValuation = computed(() => isValuationStrategy(strategy.value))
const isDividendBand = computed(() => isDividendBandStrategy(strategy.value))
const previewEnabled = computed(() => accountPreviewAvailable(strategy.value))
const previewReason = computed(() => accountPreviewReason(strategy.value))
const valuationCaveatText = computed(() => isValuation.value ? valuationCaveat(strategy.value) : '')
const holdingCountText = computed(() => isDividendBand.value
  ? `${strategy.value?.holdings?.length || 0}项指数代理`
  : `${strategy.value?.holdings?.length || 0}只ETF`)
const emptyAdviceText = computed(() => isDividendBand.value
  ? '本期隐含股息率未触发七档变化，保持当前红利、国债与现金指数等值配置。'
  : isValuation.value
    ? '本期未触发分批买入或清仓，保持当前估值账本。'
    : '本期未触发调仓，保持当前账户。')
const quantityText = row => isDividendBand.value
  ? '指数等值'
  : row.quantity == null || !Number.isFinite(Number(row.quantity))
    ? '—'
    : number(row.quantity, Number(row.quantity) % 1 ? 2 : 0)
const amountText = row => !isDividendBand.value && (row.quantity == null || Number(row.quantity) === 0)
  ? '—'
  : number(row.estimated_amount, 0)
const positionSize = holding => isDividendBand.value || holding.equivalent || holding.quantity == null
  ? '指数等值'
  : number(holding.quantity)
const adviceNote = computed(() => {
  const basis = strategy.value?.advice?.basis || '按周首收盘信号判断'
  if (isDividendBand.value) {
    const execution = executionBasisLabel(strategy.value?.advice?.execution_basis)
    return `${basis}。${execution}。金额仅为模型调仓参考，指数等值仓位不能直接作为ETF整数委托。`
  }
  return `${basis}。买入按100份取整；不预支尚未成交的卖单，开盘成交仍需校验。`
})
const previewUnavailableDetail = computed(() => {
  if (isDividendBand.value) return '红利端使用H20269全收益指数的分数等值份额，国债端使用SH511260指数代理回放。模型没有统一的ETF净值、整数份额和可成交报告价，不能把指数等值仓位直接作为券商委托。'
  if (isValuation.value) return '估值分批策略需要已持有批次、上次触发和冷却历史才能确定下一次动作。只提供现金与当前份额无法还原状态，页面不会套用通信 / 黄金策略的算法。'
  return '当前策略未声明账户试算能力，页面不会套用其他策略的计算逻辑。'
})
const universe = ref([]), universeLoading = ref(false), universeError = ref(''), cash = ref(1000000), holdings = ref([]), calculating = ref(false), calculation = ref(null), calcError = ref('')
let universeController, previewController, revision = 0, previewRevision = 0, disposed = false, nextKey = 0
function addHolding() { holdings.value.push({ key: ++nextKey, symbol: '', quantity: 100, holding_days: 0 }) }
async function fetchUniverse() {
  const current = ++revision
  universeController?.abort(); previewController?.abort(); universe.value = []; holdings.value = []; calculation.value = null; calcError.value = ''; calculating.value = false; universeError.value = ''; universeLoading.value = false
  if (!snapshot.value || !strategy.value || !previewEnabled.value) return
  universeController = new AbortController(); universeLoading.value = true
  try {
    const result = await api.get('/development/universe', { params: { strategy_id: strategyId.value, run_id: snapshot.value.run_id }, signal: universeController.signal, silentErrors: true })
    if (!disposed && current === revision) universe.value = result.rows
  } catch (err) { if (!disposed && current === revision && err.code !== 'ERR_CANCELED') universeError.value = requestError(err, '无法读取当前策略可交易标的，请刷新快照。') }
  finally { if (!disposed && current === revision) universeLoading.value = false }
}
async function calculate() {
  if (calculating.value || !strategy.value || !previewEnabled.value) return
  if (new Set(holdings.value.map(item => item.symbol)).size !== holdings.value.length) { calcError.value = '同一ETF请合并份额后输入，不能重复添加。'; return }
  const current = revision, currentInput = ++previewRevision, id = strategyId.value, runId = snapshot.value.run_id
  calculating.value = true; calcError.value = ''; calculation.value = null; previewController = new AbortController()
  try {
    const result = await api.post(`/strategies/${id}/preview`, { run_id: runId, cash: cash.value, holdings: holdings.value.map(({ symbol, quantity, holding_days }) => ({ symbol, quantity, holding_days })) }, { signal: previewController.signal, silentErrors: true })
    if (!disposed && current === revision && currentInput === previewRevision) calculation.value = result
  } catch (err) { if (!disposed && current === revision && currentInput === previewRevision && err.code !== 'ERR_CANCELED') calcError.value = requestError(err, '试算失败，请检查现金与持仓输入。') }
  finally { if (!disposed && current === revision && currentInput === previewRevision) calculating.value = false }
}
watch([strategyId, () => snapshot.value?.run_id], fetchUniverse)
watch(() => snapshot.value?.strategies, strategies => {
  if (strategies?.length && !strategies.some(item => item.id === strategyId.value)) strategyId.value = resolveStrategyId('', strategies)
}, { immediate: true })
watch([cash, holdings], () => { previewRevision++; previewController?.abort(); calculating.value = false; calculation.value = null; calcError.value = '' }, { deep: true })
onBeforeUnmount(() => { disposed = true; revision++; universeController?.abort(); previewController?.abort() })
</script>
<style scoped>
.valuation-summary { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1px; margin: 0 0 15px; border: 1px solid #e5eaf0; background: #e5eaf0; border-radius: 5px; overflow: hidden; }.valuation-summary > div { display: flex; flex-direction: column; gap: 7px; padding: 12px 14px; background: #f9fbfc; min-width: 0; }.valuation-summary span { color: var(--muted); font-size: 10px; }.valuation-summary b { color: #485f77; font-size: 11px; font-weight: 500; overflow-wrap: anywhere; }.valuation-caveat { margin: -5px 0 16px; }.preview-unavailable { background: #fafbfc; }.unavailable-reason { border-top: 1px solid var(--line); padding-top: 17px; color: var(--muted); }.unavailable-reason b { color: #52677d; font-size: 12px; font-weight: 500; }.unavailable-reason p { margin: 8px 0 0; font-size: 11px; line-height: 1.9; max-width: 780px; }
.decision-meta { display: grid; grid-template-columns: 1fr 1fr .7fr 1.2fr; gap: 18px; padding: 19px 0; margin-bottom: 15px; border-block: 1px solid var(--line); }.decision-meta > div { display: flex; flex-direction: column; gap: 9px; }.decision-meta span { color: var(--muted); font-size: 11px; }.decision-meta b { color: #425971; font-weight: 500; font-size: 13px; }.order-side { display: inline-block; padding: 5px 8px; border-radius: 4px; font-size: 11px; }.order-side.buy { background: #edf4fd; color: #2b6ad6; }.order-side.sell { background: #f8eeee; color: #ad635c; }.order-side.hold { background: #f2f5f7; color: #718394; }.decision-link { display: inline-block; margin-top: 18px; font-size: 11px; }.cash-field { max-width: 260px; }.account-input-row { display: grid; grid-template-columns: 1.6fr 1fr 1fr auto; gap: 15px; margin-top: 17px; align-items: end; }.remove-position { border: 0; background: transparent; color: #aa6d61; cursor: pointer; font-size: 11px; padding: 11px 5px; }.account-buttons { display: flex; gap: 12px; margin-top: 21px; flex-wrap: wrap; }.preview-error { margin-top: 20px; }.preview-result { border-top: 1px solid var(--line); margin-top: 24px; padding-top: 14px; }.preview-result .dev-note { margin-bottom: 15px; }
@media(max-width: 760px) { .valuation-summary { grid-template-columns: 1fr 1fr; } }
@media(max-width: 650px) { .decision-meta { grid-template-columns: 1fr 1fr; }.account-input-row { grid-template-columns: 1fr 1fr; }.account-input-row > label:first-child { grid-column: 1 / -1; }.remove-position { text-align: left; } }
@media(max-width: 460px) { .valuation-summary { grid-template-columns: 1fr; } }
</style>
