<template>
  <section :id="strategy.id" class="strategy-section" :style="{ '--accent': strategy.color }">
    <header class="strategy-heading">
      <div class="strategy-identity"><span class="strategy-number">{{ index === 0 ? '01' : '02' }}</span><div><div class="eyebrow">{{ index === 0 ? 'CORE & SATELLITE' : 'TREND & ALLOCATION' }}</div><h2>{{ strategy.name }}</h2></div></div>
      <div class="strategy-tags"><span class="tag formal">正式服务 · v{{ strategy.version }}</span><span class="tag caution">投资准入未通过</span></div>
    </header>
    <p class="strategy-description">{{ strategy.description }}</p>
    <div class="metric-grid">
      <div><span>当前模型资金</span><strong>{{ money(strategy.metrics.final_account / 10000) }}<small> 万元</small></strong><em>起始 100 万元</em></div>
      <div><span>年化收益</span><strong>{{ pct(strategy.metrics.cagr) }}</strong><em :class="{ unmet: !strategy.goals.cagr }">目标 ≥13.50% · {{ strategy.goals.cagr ? '达标' : '未达标' }}</em></div>
      <div><span>全期最大回撤</span><strong>{{ pct(strategy.metrics.max_drawdown) }}</strong><em>目标 ≤40% · {{ strategy.goals.max_drawdown ? '达标' : '未达标' }}</em></div>
      <div><span>最差完整年度</span><strong class="negative">{{ pct(strategy.metrics.worst_complete_year?.return) }}</strong><em class="unmet">{{ strategy.metrics.worst_complete_year?.year }}年 · 目标 ≥−5%</em></div>
    </div>
    <div class="advice-block">
      <div class="advice-heading"><h3>最新调仓建议 <span>模型账户</span></h3><span class="signal-state" :class="strategy.runtime.advice_status">{{ strategy.runtime.advice_label }}</span></div>
      <div class="signal-meta"><span>信号收盘 <b>{{ strategy.advice.signal_date }}</b></span><span>计划开盘 <b>{{ strategy.advice.execution_date }}</b></span><span>调仓指令 <b>{{ strategy.advice.orders_count }} 笔</b></span><span>下次周度评估 <b>{{ strategy.runtime.next_signal_date }} 19:00</b></span></div>
      <div class="table-scroll"><table class="advice-table"><thead><tr><th>ETF / 代码</th><th>动作</th><th class="numeric">份额</th><th class="numeric">参考价</th><th class="numeric">估算金额</th><th class="numeric">当前 → 目标</th><th>触发原因</th></tr></thead><tbody>
        <tr v-for="row in strategy.advice.rows" :key="row.symbol"><td><b>{{ row.name }}</b><small>{{ row.code }}</small></td><td><span class="action" :class="row.side">{{ sideName(row.side) }}</span></td><td class="numeric">{{ row.quantity ? money(row.quantity, row.quantity % 1 ? 2 : 0) : '—' }}</td><td class="numeric">{{ money(row.reference_price, 3) }}</td><td class="numeric">{{ row.quantity ? money(row.estimated_amount, 0) : '—' }}</td><td class="numeric weight-change">{{ pct(row.current_weight) }} <span>→</span> {{ pct(row.target_weight) }}</td><td class="reason">{{ row.reason }}</td></tr>
        <tr v-if="!strategy.advice.rows.length"><td colspan="7" class="empty-cell">本周无合格建仓信号，保持现金。</td></tr>
      </tbody></table></div>
      <p class="advice-note">{{ strategy.advice.basis }}。买入按100份取整，不预支尚未成交的卖单；实际账户请使用下方试算。</p>
    </div>
    <details class="holdings-details"><summary>查看当前模型持仓 <span>{{ strategy.holdings.length }}只ETF · 可用现金 {{ money(strategy.cash / 10000) }}万元</span></summary><div class="holdings-list"><div v-for="h in strategy.holdings" :key="h.symbol"><span>{{ h.name }} <small>{{ h.code }}</small></span><b>{{ money(h.quantity, 2) }}份</b><span>{{ pct(h.weight) }}</span></div><div><span>可用现金</span><b>{{ money(strategy.cash) }}元</b></div><div v-if="strategy.dividend_receivable"><span>分红应收（不可用于买入）</span><b>{{ money(strategy.dividend_receivable) }}元</b></div></div><p class="muted-note">持仓估值日 {{ strategy.data_end }}；{{ strategy.account_basis }}。</p></details>
    <WeeklyHoldings :strategy="strategy" />
    <StrategyChart :strategy="strategy" />
    <WeeklyAllocation v-if="strategy.id === 'trend200_vol15'" :strategy="strategy" />
    <details class="account-details"><summary>用我的账户试算 <span>输入信号收盘时的现金与持仓</span></summary>
      <p class="muted-note">试算基于 {{ strategy.advice.signal_date }} 收盘信号。{{ strategy.runtime.advice_label }}；结果只供模拟核对，真实账户输入不保存。</p>
      <form @submit.prevent="calculate"><div class="account-cash"><label :for="`cash-${strategy.id}`">可用现金（元）</label><input :id="`cash-${strategy.id}`" v-model.number="cash" type="number" min="0" max="1000000000000" step="0.01" required></div>
        <div v-for="(h, i) in holdings" :key="i" class="holding-input"><label>ETF代码<select v-model="h.symbol" required><option value="" disabled>选择ETF</option><option v-for="s in symbols" :key="s.symbol" :value="s.symbol">{{ s.code }} {{ s.name }}</option></select></label><label>持有份额<input v-model.number="h.quantity" type="number" min="0" max="10000000000" step="1" required></label><label>已持有交易日<input v-model.number="h.holding_days" type="number" min="0" max="10000" required></label><button type="button" class="remove-holding" @click="holdings.splice(i, 1)" :aria-label="`删除第${i + 1}行持仓`">移除</button></div>
        <div class="account-actions"><button type="button" class="secondary-button" @click="holdings.push({ symbol: '', quantity: 100, holding_days: 0 })" :disabled="holdings.length >= 29">＋ 添加持仓</button><button class="primary-button" :disabled="calculating">{{ calculating ? '计算中…' : '生成账户试算' }}</button></div>
      </form>
      <p v-if="calcError" role="alert" class="inline-error">{{ calcError }}</p>
      <div v-if="calculation" class="calculation-result"><p>输入账户估值 {{ money(calculation.account) }}元 · {{ calculation.basis }}</p><div class="table-scroll"><table><thead><tr><th>ETF</th><th>动作</th><th class="numeric">份额</th><th class="numeric">目标权重</th></tr></thead><tbody><tr v-for="r in calculation.rows" :key="r.symbol"><td>{{ r.name }} <small>{{ r.code }}</small></td><td>{{ sideName(r.side) }}</td><td class="numeric">{{ money(r.quantity, r.quantity % 1 ? 2 : 0) }}</td><td class="numeric">{{ pct(r.target_weight) }}</td></tr><tr v-if="!calculation.rows.length"><td colspan="4">无调仓指令，保持现金。</td></tr></tbody></table></div></div>
    </details>
    <footer class="strategy-footer"><span>{{ strategy.data_start }} — {{ strategy.data_end }}</span><span>{{ strategy.account_basis }}</span></footer>
  </section>
</template>

<script setup>
import { ref, computed, watch, onBeforeUnmount } from 'vue'
import api from '../api'
import StrategyChart from './StrategyChart.vue'
import WeeklyHoldings from './WeeklyHoldings.vue'
import WeeklyAllocation from './WeeklyAllocation.vue'
const props = defineProps({ strategy: { type: Object, required: true }, index: Number, universe: Array })
const money = (v, d = 2) => Number(v).toLocaleString('zh-CN', { minimumFractionDigits: d, maximumFractionDigits: d })
const pct = v => v == null ? '—' : (v * 100).toFixed(2) + '%'
const sideName = s => ({ buy: '买入', sell: '卖出', hold: '保持' })[s]
const cash = ref(1000000), holdings = ref([]), calculating = ref(false), calculation = ref(null), calcError = ref('')
const symbols = computed(() => props.universe?.filter(s => props.strategy.id === 'lightgbm' ? !['513100', '159920'].includes(s.code) : ['515880', '518880', '510300'].includes(s.code)) || [])
let previewRevision = 0, previewController
function invalidatePreview() {
  previewRevision++; previewController?.abort(); calculation.value = null; calcError.value = ''; calculating.value = false
}
watch([cash, holdings, () => props.strategy.run_id], invalidatePreview, { deep: true, flush: 'sync' })
onBeforeUnmount(invalidatePreview)
async function calculate() {
  previewController?.abort()
  const current = ++previewRevision
  previewController = new AbortController()
  calculating.value = true; calcError.value = ''; calculation.value = null
  try {
    const result = await api.post(`/strategies/${props.strategy.id}/preview`, { run_id: props.strategy.run_id, cash: cash.value, holdings: holdings.value.map(h => ({ ...h })) }, { signal: previewController.signal, silentErrors: true })
    if (current === previewRevision) calculation.value = result
  } catch (e) { if (current === previewRevision && e.code !== 'ERR_CANCELED') { const d = e.response?.data?.detail; calcError.value = Array.isArray(d) ? d.map(i => i.msg).join('；') : d || '试算失败，请检查输入' } }
  finally { if (current === previewRevision) calculating.value = false }
}
</script>

<style scoped>
.strategy-section { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 32px; scroll-margin-top: 90px; margin-bottom: 28px; box-shadow: 0 3px 15px #16283e03; }
.strategy-heading, .strategy-identity, .advice-heading { display: flex; align-items: center; justify-content: space-between; gap: 15px; }
.strategy-identity { justify-content: flex-start; gap: 18px; }.strategy-number { color: var(--accent); font-size: 28px; font-family: Georgia, serif; padding-right: 18px; border-right: 1px solid var(--line); }
.eyebrow { color: var(--accent); letter-spacing: 1.9px; font-size: 10px; margin-bottom: 6px; } h2 { font-size: 23px; letter-spacing: -.6px; margin: 0; font-weight: 600; }
.strategy-tags { display: flex; gap: 8px; }.tag { font-size: 11px; padding: 6px 9px; border: 1px solid var(--line); border-radius: 4px; white-space: nowrap; }.formal { color: var(--accent); }.caution { color: #95733d; background: #faf7f0; border-color: #eadfcd; }
.strategy-description { font-size: 13px; color: var(--muted); margin: 20px 0 26px; line-height: 1.9; }
.metric-grid { display: grid; grid-template-columns: 1.25fr 1fr 1fr 1fr; background: #f8fafb; border: 1px solid #edf0f3; padding: 22px; border-radius: 7px; gap: 18px; }.metric-grid > div { display: flex; flex-direction: column; gap: 10px; }.metric-grid > div + div { padding-left: 20px; border-left: 1px solid var(--line); }.metric-grid span { font-size: 12px; color: var(--muted); }.metric-grid strong { font-family: 'Bahnschrift', 'Arial', sans-serif; font-size: 29px; font-weight: 500; font-variant-numeric: tabular-nums; letter-spacing: -.8px; }.metric-grid small { font-size: 11px; color: var(--muted); letter-spacing: 0; }.metric-grid em { font-size: 11px; color: var(--muted); font-style: normal; }.metric-grid .unmet { color: #9a7342; }.negative { color: #ae6058; }
.advice-block { margin-top: 30px; }.advice-heading h3 { font-size: 17px; margin: 0; }.advice-heading h3 span { font-size: 11px; color: var(--muted); margin-left: 10px; font-weight: 400; }.signal-state { font-size: 11px; color: #627688; padding: 6px 9px; background: #f0f4f7; border-radius: 4px; }.signal-state.stale { color: #aa5c48; background: #faf0ea; }.signal-state.pending { color: #3e8163; background: #eef7f1; }
.signal-meta { display: flex; gap: 23px; margin: 17px 0; font-size: 11px; color: var(--muted); flex-wrap: wrap; }.signal-meta b { color: #42536a; font-weight: 500; margin-left: 5px; }
.table-scroll { overflow-x: auto; } table { width: 100%; border-collapse: collapse; font-size: 12px; } th { text-align: left; color: #7d8995; font-weight: 500; padding: 12px 10px; background: #f7f9fa; border-block: 1px solid var(--line); white-space: nowrap; } td { padding: 14px 10px; border-bottom: 1px solid #edf0f3; white-space: nowrap; } td b { font-weight: 500; } td small { display: block; color: #8995a3; font-size: 10px; margin-top: 5px; letter-spacing: .5px; }.numeric { text-align: right; font-variant-numeric: tabular-nums; }.weight-change span { color: #a3abb5; margin: 0 4px; }.reason { color: var(--muted); font-size: 11px; white-space: normal; min-width: 135px; line-height: 1.7; max-width: 200px; }.action { padding: 4px 8px; border-radius: 4px; font-size: 11px; }.buy { background: #ecf3fd; color: #2b6ad6; }.sell { background: #f8eeed; color: #ab625a; }.hold { background: #f2f4f5; color: #7e8993; }.empty-cell { text-align: center; padding: 25px; color: var(--muted); }
.advice-note, .muted-note { color: var(--muted); font-size: 11px; line-height: 1.9; margin: 12px 0 0; }
details { margin-top: 22px; border: 1px solid var(--line); border-radius: 6px; padding: 14px 16px; } summary { font-size: 12px; font-weight: 500; cursor: pointer; } summary span { font-weight: 400; color: var(--muted); font-size: 11px; margin-left: 12px; }.holdings-list { display: grid; gap: 10px; margin-top: 20px; font-size: 12px; }.holdings-list > div { display: grid; grid-template-columns: 1fr 1fr 70px; }.holdings-list b { font-weight: 500; text-align: right; }.holdings-list > div > span:last-child { text-align: right; }.holdings-list small { color: var(--muted); margin-left: 10px; }
form { margin-top: 20px; } label { display: flex; flex-direction: column; gap: 7px; font-size: 11px; color: var(--muted); } input, select { padding: 9px 11px; border: 1px solid #d8dfe6; border-radius: 5px; font-size: 12px; background: white; color: #233448; min-width: 0; } .account-cash { max-width: 260px; }.account-cash input { width: 100%; box-sizing: border-box; margin-top: 8px; }.holding-input { display: grid; grid-template-columns: 1.5fr 1fr 1fr auto; align-items: end; gap: 12px; margin-top: 14px; }.remove-holding { background: none; border: 0; font-size: 11px; color: #9e655d; cursor: pointer; padding: 10px; }.account-actions { display: flex; gap: 12px; margin-top: 18px; }.primary-button, .secondary-button { border-radius: 5px; padding: 10px 15px; font-size: 12px; cursor: pointer; }.primary-button { background: #21364f; color: white; border: 1px solid #21364f; }.secondary-button { border: 1px solid var(--line); background: white; color: #536579; } button:disabled { opacity: .5; cursor: wait; }.inline-error { color: #ac5c4f; font-size: 12px; }.calculation-result { margin-top: 20px; }.calculation-result p { font-size: 11px; color: var(--muted); line-height: 1.8; }.strategy-footer { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; font-size: 10px; color: #8995a3; border-top: 1px solid var(--line); margin-top: 24px; padding-top: 17px; }
@media(max-width: 850px) { .strategy-section { padding: 22px; }.strategy-heading { align-items: flex-start; flex-wrap: wrap; }.metric-grid { grid-template-columns: 1fr 1fr; }.metric-grid > div:nth-child(3) { padding-left: 0; border-left: 0; } .metric-grid strong { font-size: 26px; } }
@media(max-width: 520px) { .strategy-section { padding: 18px 15px; border-radius: 8px; } h2 { font-size: 20px; }.strategy-number { font-size: 24px; }.metric-grid { padding: 16px; gap: 22px 10px; }.metric-grid > div + div { padding-left: 14px; }.metric-grid strong { font-size: 23px; }.advice-heading { flex-wrap: wrap; }.signal-meta { gap: 12px; } summary span { display: block; margin: 10px 0 0; }.holding-input { grid-template-columns: 1fr 1fr; gap: 12px; }.holding-input > label:first-child { grid-column: 1 / -1; }.remove-holding { text-align: left; } }
</style>
