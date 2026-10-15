<template>
  <DevelopmentFrame title="选ETF清单" subtitle="按正式周首信号查看候选。LightGBM显示真实模型排名与独立底仓；通信 / 黄金显示市场资格及200日趋势门槛。" :snapshot="snapshot" :loading="loading" :error="error" @reload="load">
    <section class="dev-panel"><DevStrategySelector v-model="strategyId" :strategies="snapshot.strategies" /><form class="dev-toolbar picks-controls" @submit.prevent="query()"><label>周首信号日<select v-model="date"><option value="">最新信号</option><option v-for="day in dates" :key="day" :value="day">{{ day }}</option></select></label><label v-if="strategyId === 'lightgbm'">显示排名数<input v-model.number="topk" type="number" min="1" max="27" required></label><button class="dev-button primary" :disabled="querying">{{ querying ? '查询中…' : '查询清单' }}</button><button type="button" class="dev-button" :disabled="querying" @click="latest">最新信号</button></form></section>
    <div v-if="queryError" class="dev-notice error" role="alert">{{ queryError }}</div>
    <div v-if="querying" class="dev-loading" role="status"><span class="dev-spinner"></span>正在读取该周正式候选…</div>
    <template v-else-if="result">
      <div v-if="result.fallback" class="dev-notice warning">请求 {{ result.requested_date }}，已回退至可用周首信号 {{ result.date }}。</div>
      <section class="dev-panel"><div class="dev-panel-head"><div><h2>{{ result.kind === 'rank' ? '权益卫星 · 模型排名' : '固定配置 · 资格与趋势' }}</h2><p>{{ result.description }}</p></div><span>信号 {{ result.date }}</span></div><div class="dev-table-scroll"><table class="dev-table"><thead><tr><th v-if="result.kind === 'rank'" class="numeric">排名</th><th>ETF / 代码</th><th>市场资格</th><th v-if="result.kind === 'rank'" class="numeric">模型得分</th><th class="numeric">收盘价</th><template v-if="result.kind === 'trend'"><th class="numeric">200日均线</th><th class="numeric">相对均线</th><th>趋势门槛</th></template><th>判断原因</th></tr></thead><tbody><tr v-for="row in result.rows" :key="row.symbol"><td v-if="result.kind === 'rank'" class="numeric">{{ number(row.rank, 0) }}</td><td>{{ row.name }}<small>{{ row.code }}</small></td><td><span class="dev-badge" :class="row.eligible ? 'pass' : 'fail'">{{ row.eligible ? '合格' : '不合格' }}</span></td><td v-if="result.kind === 'rank'" class="numeric">{{ number(row.score, 6) }}</td><td class="numeric">{{ number(row.close, 3) }}</td><template v-if="result.kind === 'trend'"><td class="numeric">{{ number(row.sma200, 4) }}</td><td class="numeric" :class="{ 'dev-negative': row.deviation < 0 }">{{ percent(row.deviation) }}</td><td><span class="dev-badge" :class="row.trend_pass ? 'pass' : 'fail'">{{ row.trend_pass == null ? '数据不足' : row.trend_pass ? '通过' : '未通过' }}</span></td></template><td class="reason">{{ row.reason }}</td></tr><tr v-if="!result.rows.length"><td :colspan="result.kind === 'rank' ? 6 : 8" class="dev-empty">本周没有可显示的候选，策略保持现金或已有合格持仓。</td></tr></tbody></table></div><p class="dev-note">{{ result.kind === 'rank' ? '模型分数用于相对排序；买卖还需满足最小持有期、排名缓冲、风控、现金与成交约束。' : '均线用复权收盘价计算，再换算为当日原价等值。通信跌破均线时持现金，只有市场资格失效时才允许沪深300替代。' }}</p></section>
      <section v-if="result.kind === 'rank'" class="dev-panel"><div class="dev-panel-head"><div><h2>国债 / 黄金 · 独立底仓</h2><p>底仓由正式配置给定，不参与权益ETF模型排名。</p></div></div><div class="dev-table-scroll"><table class="dev-table"><thead><tr><th>ETF / 代码</th><th>市场资格</th><th class="numeric">收盘价</th><th>配置说明</th></tr></thead><tbody><tr v-for="row in result.core_rows" :key="row.symbol"><td>{{ row.name }}<small>{{ row.code }}</small></td><td><span class="dev-badge" :class="row.eligible ? 'pass' : 'fail'">{{ row.eligible ? '合格' : '不合格' }}</span></td><td class="numeric">{{ number(row.close, 3) }}</td><td class="reason">{{ row.reason }}</td></tr><tr v-if="!result.core_rows?.length"><td colspan="4" class="dev-empty">暂无底仓信息。</td></tr></tbody></table></div></section>
    </template>
    <div v-else-if="!querying && !queryError" class="dev-empty">暂无可用候选清单，请到数据管道重建正式快照。</div>
  </DevelopmentFrame>
</template>
<script setup>
import { ref, watch, onBeforeUnmount } from 'vue'
import { useRoute } from 'vue-router'
import api from '../api'
import DevelopmentFrame from '../components/DevelopmentFrame.vue'
import DevStrategySelector from '../components/DevStrategySelector.vue'
import { useFormalSnapshot, requestError, number, percent } from '../composables/useFormalSnapshot'
const route = useRoute()
const { snapshot, loading, error, load } = useFormalSnapshot()
const strategyId = ref(route.query.strategy === 'trend200_vol15' ? 'trend200_vol15' : 'lightgbm')
const date = ref(''), dates = ref([]), topk = ref(6), result = ref(null), querying = ref(false), queryError = ref('')
let controller, revision = 0, disposed = false
async function query(resetDates = false) {
  if (!snapshot.value) return
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
    const data = await api.get('/development/picks', { ...options, params: { strategy_id: id, date: date.value || undefined, topk: topk.value, run_id: runId } })
    if (!disposed && current === revision) result.value = data
  } catch (err) { if (!disposed && current === revision && err.code !== 'ERR_CANCELED') queryError.value = requestError(err, '候选查询失败，请刷新快照后重试。') }
  finally { if (!disposed && current === revision) querying.value = false }
}
function latest() { date.value = ''; query() }
watch([strategyId, () => snapshot.value?.run_id], () => { date.value = ''; dates.value = []; query(true) })
onBeforeUnmount(() => { disposed = true; revision++; controller?.abort() })
</script>
<style scoped>.picks-controls { margin-top: 23px; border-top: 1px solid var(--line); padding-top: 20px; }</style>
