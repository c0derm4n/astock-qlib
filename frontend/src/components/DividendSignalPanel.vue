<template>
  <section class="dividend-band-panel" aria-label="隐含股息率七档估值信号">
    <div class="band-heading">
      <div class="yield-signal">
        <span>周首隐含股息率</span>
        <strong>{{ percent(signal.implied_yield) }}</strong>
        <small>{{ signal.date || strategy?.advice?.signal_date || '—' }}</small>
      </div>
      <div class="band-basis">
        <b>红利低波七档估值带</b>
        <span>价格指数 {{ signal.price_index || 'H30269' }} · 全收益指数 {{ signal.return_index || 'H20269' }}</span>
        <p>按周首收盘信号判断；执行口径为{{ executionText }}，回放仅作模型核对。</p>
      </div>
    </div>
    <div class="allocation-grid">
      <article v-for="row in allocationRows" :key="row.asset" class="allocation-row">
        <div class="asset-name"><b>{{ row.name }}</b><small>{{ row.symbol }}</small></div>
        <div class="weight-flow" :aria-label="`${row.name}仓位`">
          <div><small>当前</small><strong>{{ showCurrent ? percent(row.currentWeight) : '—' }}</strong></div>
          <span aria-hidden="true">→</span>
          <div><small>目标</small><strong>{{ percent(row.targetWeight) }}</strong></div>
        </div>
        <div class="model-amount">
          <span>{{ showCurrent ? `模型金额 ${money(row.currentAmount)} 元` : '目标仓位' }}</span>
          <span>目标金额 {{ money(row.targetAmount) }} 元</span>
        </div>
      </article>
    </div>
    <p v-if="decisionText" class="decision-detail">{{ decisionText }}</p>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import { percent } from '../composables/useFormalSnapshot'
import { executionBasisLabel } from '../strategyPresentation'

const props = defineProps({
  strategy: { type: Object, default: null },
  signal: { type: Object, default: undefined },
  decisionDetail: { type: [String, Object, Array], default: null },
  showCurrent: { type: Boolean, default: true },
})

const signal = computed(() => props.signal ?? props.strategy?.dividend_signal ?? {})
const executionText = computed(() => executionBasisLabel(props.strategy?.advice?.execution_basis))
const nav = computed(() => {
  const finalAccount = Number(props.strategy?.metrics?.final_account)
  if (Number.isFinite(finalAccount) && finalAccount > 0) return finalAccount
  const holdingsValue = (props.strategy?.holdings || []).reduce((sum, item) => sum + (Number(item.value) || 0), 0)
  return holdingsValue + (Number(props.strategy?.cash) || 0)
})

const holdingFor = asset => (props.strategy?.holdings || []).find(item => item.asset === asset
  || (asset === 'red' && item.symbol === 'H20269')
  || (asset === 'bond' && item.symbol === 'SH511260'))

const currentWeight = asset => {
  if (asset === 'cash') {
    const value = Number(props.strategy?.cash)
    if (Number.isFinite(value) && nav.value > 0) return value / nav.value
    const invested = (props.strategy?.holdings || []).reduce((sum, item) => sum + (Number(item.weight) || 0), 0)
    return invested > 0 ? Math.max(0, 1 - invested) : null
  }
  const holding = holdingFor(asset)
  if (!holding && nav.value > 0 && Array.isArray(props.strategy?.holdings)) return 0
  return holding?.weight == null ? null : Number(holding.weight)
}

const currentAmount = asset => {
  if (asset === 'cash') return Number.isFinite(Number(props.strategy?.cash)) ? Number(props.strategy.cash) : null
  const holding = holdingFor(asset)
  if (holding?.value != null && Number.isFinite(Number(holding.value))) return Number(holding.value)
  const weight = currentWeight(asset)
  return weight == null || !(nav.value > 0) ? null : weight * nav.value
}

const targetAmount = weight => weight == null || !(nav.value > 0) ? null : Number(weight) * nav.value

const allocationRows = computed(() => [
  { asset: 'red', name: '红利', symbol: signal.value.return_index || 'H20269', targetWeight: signal.value.red_target },
  { asset: 'bond', name: '国债', symbol: 'SH511260', targetWeight: signal.value.bond_target },
  { asset: 'cash', name: '现金', symbol: 'CASH', targetWeight: signal.value.cash_target },
].map(row => ({
  ...row,
  currentWeight: currentWeight(row.asset),
  currentAmount: currentAmount(row.asset),
  targetAmount: targetAmount(row.targetWeight),
})))

const decisionText = computed(() => {
  const detail = props.decisionDetail ?? props.strategy?.advice?.decision_detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.filter(Boolean).join('；')
  if (detail && typeof detail === 'object') {
    const text = ['summary', 'text', 'description', 'reason', 'band']
      .map(key => detail[key])
      .filter(value => typeof value === 'string' && value)
      .join('；')
    if (text) return text
  }
  const current = signal.value
  if (current.implied_yield == null && current.red_target == null) return ''
  return `隐含股息率${percent(current.implied_yield)}；红利${percent(current.red_target)}、国债${percent(current.bond_target)}、现金${percent(current.cash_target)}`
})

const money = value => value == null || !Number.isFinite(Number(value))
  ? '—'
  : Number(value).toLocaleString('zh-CN', { maximumFractionDigits: 0 })
</script>

<style scoped>
.dividend-band-panel { margin: 0 0 24px; padding: 18px 20px; border: 1px solid #eadfe2; border-radius: 7px; background: linear-gradient(135deg, #fdfbfb 0%, #f8fafc 100%); }
.band-heading { display: flex; align-items: center; justify-content: space-between; gap: 20px; }
.yield-signal { display: grid; grid-template-columns: auto auto; align-items: baseline; gap: 2px 12px; min-width: 190px; }
.yield-signal span { grid-column: 1 / -1; color: var(--muted); font-size: 10px; }
.yield-signal strong { color: var(--accent, #a84c54); font: 500 26px 'Bahnschrift', sans-serif; font-variant-numeric: tabular-nums; }
.yield-signal small { color: var(--muted); font-size: 10px; }
.band-basis { text-align: right; color: var(--muted); font-size: 10px; line-height: 1.8; }
.band-basis b, .band-basis span { display: block; }
.band-basis b { color: #526579; font-size: 13px; font-weight: 500; }
.band-basis p { margin: 2px 0 0; }
.allocation-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); margin-top: 16px; border: 1px solid #e7ebef; border-radius: 5px; overflow: hidden; background: #e7ebef; gap: 1px; }
.allocation-row { display: grid; gap: 9px; min-width: 0; padding: 13px 15px; background: #fff; }
.asset-name, .model-amount { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.asset-name b { color: #42566e; font-size: 13px; font-weight: 500; }
.asset-name small { color: var(--muted); font-size: 9px; }
.weight-flow { display: flex; align-items: flex-end; gap: 10px; }
.weight-flow > div { display: flex; flex-direction: column; gap: 2px; }
.weight-flow small { color: #9aa4af; font-size: 8px; }
.weight-flow strong { color: #344b63; font: 500 18px 'Bahnschrift', sans-serif; font-variant-numeric: tabular-nums; }
.weight-flow > div:last-child strong { color: var(--accent, #a84c54); }
.weight-flow span { color: #a1aab4; font-size: 12px; }
.model-amount { color: var(--muted); font-size: 9px; }
.model-amount span:last-child { color: #5e7286; }
.decision-detail { margin: 12px 0 0; padding-top: 11px; border-top: 1px solid #e8edf1; color: #65778b; font-size: 10px; line-height: 1.8; }
@media(max-width: 760px) { .allocation-grid { grid-template-columns: 1fr; }.band-heading { align-items: flex-start; } }
@media(max-width: 520px) { .dividend-band-panel { padding: 15px 13px; }.band-heading { flex-direction: column; gap: 8px; }.band-basis { text-align: left; }.yield-signal strong { font-size: 23px; } }
</style>
