<template>
  <div class="weekly-history">
    <div class="history-heading"><h3>历史持仓 <span>最近{{ history.length }}周</span></h3><p>{{ historySubtitle }}</p></div>
    <p v-if="!history.length" class="history-unavailable">暂无历史持仓快照，请点击页面顶部“更新行情与建议”生成。</p>
    <div v-else class="history-table-scroll">
      <table class="history-table"><thead><tr><th>周次 / 估值日</th><th>{{ positionHeading }}</th><th class="numeric">账户资金</th><th class="numeric">可用现金</th></tr></thead>
        <tbody v-for="week in history" :key="week.week_start">
          <tr><td class="week-cell"><b>{{ week.week_start }}</b><small v-if="week.week_end !== week.week_start">至 {{ shortDate(week.week_end) }}</small><small v-else>周首信号</small><span class="week-label" v-if="week.partial">本周截至 {{ shortDate(week.date) }}</span><small v-else-if="week.carried_over">整周休市 · 沿用 {{ shortDate(week.date) }}</small><small v-else>估值 {{ shortDate(week.date) }}</small><span v-if="isDividendBand && week.dividend_signal" class="week-label">隐含股息率 {{ pct(week.dividend_signal.implied_yield) }}</span></td>
            <td class="position-cell"><div v-if="isEmptyWeek(week)" class="empty-position"><span>空仓</span><small>{{ emptyPositionText }}</small></div><div v-else class="position-tags"><span v-for="h in week.holdings" :key="h.asset || h.symbol" class="position-tag"><span>{{ h.name }} <small>{{ h.code || h.symbol }}</small></span><b>{{ pct(h.weight) }}</b></span></div><button class="detail-toggle" @click="toggle(week.week_start)" :aria-expanded="expanded.has(week.week_start)" :aria-controls="`${strategy.id}-${week.week_start}-holdings`">{{ expanded.has(week.week_start) ? '收起明细' : '查看明细' }} <span aria-hidden="true">{{ expanded.has(week.week_start) ? '−' : '＋' }}</span></button></td>
            <td class="numeric fund-cell">{{ money(week.account / 10000) }}<small>万元</small></td><td class="numeric fund-cell">{{ money(week.cash / 10000) }}<small>万元 · {{ pct(week.cash_weight) }}</small><small v-if="week.dividend_receivable">分红应收 {{ money(week.dividend_receivable) }}元</small></td>
          </tr>
          <tr v-if="expanded.has(week.week_start)" class="history-detail"><td colspan="4"><div :id="`${strategy.id}-${week.week_start}-holdings`" class="detail-content"><p>{{ week.date }} 收盘持仓 · {{ strategy.account_basis }}</p><table><thead><tr><th>标的 / 代码</th><th class="numeric">{{ isDividendBand ? '代理口径' : '持有份额' }}</th><th class="numeric">{{ isDividendBand ? '模拟金额（元）' : '市值（元）' }}</th><th class="numeric">仓位</th></tr></thead><tbody><tr v-for="h in week.holdings" :key="h.asset || h.symbol"><td>{{ h.name }} <small>{{ h.code || h.symbol }}</small></td><td class="numeric">{{ positionSize(h) }}</td><td class="numeric">{{ money(h.value) }}</td><td class="numeric">{{ pct(h.weight) }}</td></tr><tr v-if="isEmptyWeek(week)"><td colspan="4" class="detail-empty">空仓，持仓市值为0。</td></tr><tr><td>可用现金</td><td></td><td class="numeric">{{ money(week.cash) }}</td><td class="numeric">{{ pct(week.cash_weight) }}</td></tr><tr v-if="week.dividend_receivable"><td>分红应收（尚不可用）</td><td></td><td class="numeric">{{ money(week.dividend_receivable) }}</td><td class="numeric">{{ pct(week.receivable_weight) }}</td></tr></tbody></table></div></td></tr>
        </tbody>
      </table>
    </div>
    <p class="history-note">{{ historyNote }}</p>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { isDividendBandStrategy } from '../strategyPresentation'
const props = defineProps({ strategy: { type: Object, required: true } })
const history = computed(() => (props.strategy.weekly_holdings || []).map((week, index) => {
  const date = week.date || week.week_start || ''
  const account = Number(week.account)
  const cash = Number(week.cash)
  return {
    ...week,
    date,
    week_start: week.week_start || date || `week-${index}`,
    week_end: week.week_end || date,
    cash_weight: week.cash_weight ?? (Number.isFinite(account) && account > 0 && Number.isFinite(cash) ? cash / account : null),
    receivable_weight: week.receivable_weight ?? (Number.isFinite(account) && account > 0 ? (Number(week.dividend_receivable) || 0) / account : null),
    empty: week.empty ?? !(week.holdings || []).length,
  }
}))
const isDividendBand = computed(() => isDividendBandStrategy(props.strategy))
const historySubtitle = computed(() => isDividendBand.value
  ? '最近50个周首信号日的指数等值回放账本'
  : '每周最后一个交易日收盘后的真实回放账本')
const positionHeading = computed(() => isDividendBand.value ? '指数代理持仓 / 仓位' : '持仓ETF / 仓位')
const emptyPositionText = computed(() => isDividendBand.value ? '未持有指数代理仓位' : '未持有ETF')
const historyNote = computed(() => isDividendBand.value
  ? '红利与国债仓位来自周首信号日收盘的指数代理、分数等值账户回放；建议在下一交易日收盘模拟执行。页面只展示模型金额和权重，不表示实际ETF份额或成交。基础情景与代理口径不构成收益保证。'
  : '持仓来自策略真实回放账本，按已成交记录与各周收盘估值计算；本周仅展示截至行情截止日的持仓。空仓周保留，数据未覆盖的未来成交不计入。')
const expanded = ref(new Set())
const money = v => v == null || !Number.isFinite(Number(v)) ? '—' : Number(v).toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
const pct = v => v == null || !Number.isFinite(Number(v)) ? '—' : (Number(v) * 100).toFixed(2) + '%'
const shortDate = value => value ? String(value).slice(5) : '—'
const isEmptyWeek = week => week.empty || !(week.holdings || []).length
const positionSize = holding => isDividendBand.value || holding.equivalent || holding.quantity == null
  ? '指数等值'
  : money(holding.quantity)
function toggle(key) { const next = new Set(expanded.value); next.has(key) ? next.delete(key) : next.add(key); expanded.value = next }
watch(() => props.strategy.run_id, () => { expanded.value = new Set() })
</script>

<style scoped>
.weekly-history { margin-top: 30px; padding-top: 24px; border-top: 1px solid var(--line); }
.history-heading { display: flex; align-items: center; justify-content: space-between; gap: 15px; margin-bottom: 17px; } h3 { font-size: 17px; margin: 0; font-weight: 600; } h3 span { font-size: 11px; font-weight: 400; color: var(--muted); margin-left: 10px; } .history-heading p { font-size: 11px; color: var(--muted); margin: 0; }
.history-table-scroll { overflow-x: auto; } table { border-collapse: collapse; width: 100%; font-size: 12px; } .history-table { min-width: 770px; } th { text-align: left; background: #f7f9fa; color: #7d8995; border-block: 1px solid var(--line); padding: 12px 10px; font-weight: 500; white-space: nowrap; } td { border-bottom: 1px solid #edf0f3; padding: 14px 10px; vertical-align: top; } .numeric { text-align: right; font-variant-numeric: tabular-nums; } .week-cell { width: 132px; white-space: nowrap; } .week-cell b { font-weight: 500; } small { color: #8995a3; font-size: 10px; } .week-cell > small, .fund-cell > small { display: block; margin-top: 5px; } .week-label { display: inline-block; font-size: 10px; color: var(--accent); margin-top: 7px; padding: 3px 6px; background: #f1f5fa; border-radius: 3px; }
.position-cell { min-width: 370px; } .position-tags { display: flex; flex-wrap: wrap; gap: 7px; } .position-tag { display: flex; align-items: center; justify-content: space-between; gap: 10px; border: 1px solid #e7edf2; border-radius: 4px; padding: 5px 8px; background: #fbfcfd; font-size: 11px; } .position-tag small { margin-left: 5px; font-size: 9px; } .position-tag b { color: var(--accent); font-weight: 500; font-variant-numeric: tabular-nums; } .fund-cell { white-space: nowrap; min-width: 100px; } .empty-position { display: flex; align-items: center; gap: 10px; } .empty-position > span { background: #f0f3f5; color: #6b7e90; font-size: 11px; border-radius: 4px; padding: 5px 9px; }
.detail-toggle { border: 0; padding: 0; background: none; color: #7c8fa2; font-size: 10px; cursor: pointer; margin-top: 10px; } .detail-toggle:hover { color: var(--accent); } .detail-toggle span { margin-left: 4px; } .history-detail > td { background: #f8fafc; padding: 14px 20px 18px; } .detail-content p { color: var(--muted); font-size: 11px; margin: 0 0 12px; } .detail-content table { font-size: 11px; } .detail-content th { background: transparent; padding: 9px 10px; } .detail-content td { padding: 10px; } .detail-content td small { margin-left: 8px; } .detail-empty { color: var(--muted); text-align: center; }
.history-note, .history-unavailable { color: var(--muted); font-size: 11px; line-height: 1.9; margin: 12px 0 0; } .history-unavailable { padding: 20px; background: #f8fafb; border-radius: 5px; }
@media(max-width: 680px) { .history-heading { flex-direction: column; align-items: flex-start; gap: 9px; } }
</style>
