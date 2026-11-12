<template>
  <DevelopmentFrame title="数据管道" subtitle="统一更新正式策略：行情校验、按策略执行的规则或指数、周频回放与最新建议一起构建，完成后发布新快照。" :snapshot="snapshot" :loading="loading" :error="error" allow-empty @reload="load">
    <template #actions><button class="dev-button primary" :disabled="job.running || submitting || !statusReady" @click="run(true)">{{ submitting || job.running ? '任务正在运行…' : '更新行情并重建正式策略' }}</button></template>
    <div v-if="statusError" class="dev-notice error" role="alert">{{ statusError }}<button class="dev-button" @click="poll">重新连接</button></div>
    <div v-if="submitError || job.error" class="dev-notice error" role="alert">{{ submitError || job.error }}。已完成的正式快照继续保留。</div>
    <section class="dev-panel"><div class="dev-panel-head"><div><h2>正式策略构建</h2><p>两种入口都会完整生成各策略的规则、历史回放、候选与调仓建议；整个任务只能同时运行一份。</p></div><span class="dev-badge" :class="job.error ? 'fail' : job.finished_at && !job.running ? 'pass' : ''">{{ job.running ? '运行中' : job.error ? '上次任务有错误' : job.finished_at ? '已完成' : '等待任务' }}</span></div>
      <ol class="formal-phases"><li v-for="(stage, index) in stages" :key="stage.id" :class="{ active: job.running && currentStage === stage.id }"><span>{{ String(index + 1).padStart(2, '0') }}</span><div><b>{{ stage.name }}</b><p>{{ stage.description }}</p></div></li></ol>
      <div class="pipeline-actions"><button class="dev-button primary" :disabled="job.running || submitting || !statusReady" @click="run(true)">更新行情并重建正式策略</button><button class="dev-button" :disabled="job.running || submitting || !statusReady" @click="run(false)">重建规则、回放和参考</button><p>前者抓取并校验增量行情；后者使用已有正式数据库，适合核对规则或重新生成结果。</p></div>
      <div class="job-status" aria-live="polite"><div v-if="job.running"><span class="dev-spinner"></span><b>{{ stageLabel }}</b><span>{{ job.log_tail?.at(-1) || '正在启动正式构建…' }}</span></div><p v-else>{{ job.finished_at ? `最近任务结束：${formatTime(job.finished_at)}` : '当前没有运行中的正式构建任务。' }}</p><p v-if="job.started_at">开始 {{ formatTime(job.started_at) }}</p></div>
    </section>
    <section class="dev-panel"><div class="dev-panel-head"><div><h2>当前发布快照</h2><p>界面与建议始终读取已完成的快照；构建中的中间文件不对外展示。</p></div><button class="dev-button" :disabled="loading" @click="load">刷新发布状态</button></div><div v-if="snapshot" class="published-snapshot"><div><span>运行标识</span><b>{{ snapshot.run_id }}</b></div><div><span>行情截止</span><b>{{ snapshot.data_end }}</b></div><div><span>完成时间</span><b>{{ formatTime(snapshot.completed_at) }}</b></div><div><span>正式策略</span><b>{{ snapshot.strategies.map(item => item.name).join(' / ') }}</b></div></div><div v-else class="dev-empty">尚无可用正式快照，请启动完整构建。</div></section>
    <section class="dev-panel"><div class="dev-panel-head"><div><h2>运行日志</h2><p>显示本次任务最近120行；错误与行情截止日期需一起核对。</p></div><span>{{ job.running ? '自动更新' : '任务日志' }}</span></div><pre ref="logElement" class="formal-log" aria-label="正式策略构建日志">{{ job.log_tail?.join('\n') || '当前暂无任务日志。' }}</pre></section>
  </DevelopmentFrame>
</template>
<script setup>
import { computed, nextTick, onMounted, onBeforeUnmount, ref } from 'vue'
import api from '../api'
import DevelopmentFrame from '../components/DevelopmentFrame.vue'
import { useFormalSnapshot, requestError, formatTime } from '../composables/useFormalSnapshot'
const { snapshot, loading, error, load } = useFormalSnapshot()
const job = ref({ running: false, log_tail: [] }), submitting = ref(false), submitError = ref(''), statusError = ref(''), statusReady = ref(false), logElement = ref(null)
const stages = [{ id: 'fetch', name: '行情校验', description: '核对历史重叠、复权与增量数据' }, { id: 'model', name: '策略计算', description: 'PE估值、固定趋势与隐含股息率七档' }, { id: 'backtest', name: '周频回放', description: '真实份额账本与指数等值模拟' }, { id: 'publish', name: '发布建议', description: '校验完整快照后统一发布' }]
const currentStage = computed(() => ({ train: 'model', decide: 'publish', replay: 'backtest', loading: 'fetch' })[job.value.phase || job.value.current_step] || job.value.phase || job.value.current_step)
const stageLabel = computed(() => stages.find(stage => stage.id === currentStage.value)?.name || '正在构建正式策略')
let timer, controller, polling = null, wasRunning = false, disposed = false
async function poll() {
  if (disposed) return
  if (polling) return polling
  clearTimeout(timer); controller = new AbortController()
  polling = (async () => {
    try {
      const result = await api.get('/strategies/refresh/status', { signal: controller.signal, silentErrors: true })
      if (disposed) return
      job.value = result; statusError.value = ''; statusReady.value = true
      if (wasRunning && !result.running) await load()
      wasRunning = result.running
      const scroll = !logElement.value || logElement.value.scrollHeight - logElement.value.scrollTop - logElement.value.clientHeight < 60
      await nextTick()
      if (scroll && logElement.value) logElement.value.scrollTop = logElement.value.scrollHeight
    } catch (err) { if (!disposed && err.code !== 'ERR_CANCELED') { statusError.value = requestError(err, '暂时无法读取任务状态，正在重试。'); statusReady.value = false } }
    finally { polling = null; if (!disposed) timer = setTimeout(poll, job.value.running ? 2000 : 7000) }
  })()
  return polling
}
async function run(fetch) {
  if (job.value.running || submitting.value || !statusReady.value) return
  submitting.value = true; submitError.value = ''
  try {
    const result = await api.post('/strategies/refresh', { fetch }, { silentErrors: true })
    if (disposed) return
    job.value = result; wasRunning = true
    if (polling) await polling
    await poll()
  } catch (err) { if (!disposed) submitError.value = requestError(err, '正式任务未能启动，请稍后重试。') }
  finally { if (!disposed) submitting.value = false }
}
onMounted(poll)
onBeforeUnmount(() => { disposed = true; clearTimeout(timer); controller?.abort() })
</script>
<style scoped>
.formal-phases { display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; list-style: none; margin: 22px 0; padding: 0; }.formal-phases li { display: flex; gap: 12px; border-top: 2px solid #e5ebf0; padding-top: 20px; }.formal-phases li.active { border-color: #46779f; }.formal-phases li > span { font-family: Georgia, serif; font-size: 24px; color: #b0bdcb; }.formal-phases li.active > span { color: #46779f; }.formal-phases b { font-size: 13px; font-weight: 500; }.formal-phases p { color: var(--muted); font-size: 11px; line-height: 1.8; margin: 9px 0 0; }.pipeline-actions { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; border-top: 1px solid var(--line); padding-top: 23px; }.pipeline-actions p { flex-basis: 100%; color: var(--muted); font-size: 11px; margin: 0; line-height: 1.9; }.job-status { background: #f7f9fb; border: 1px solid #e9edf2; border-radius: 5px; padding: 16px; margin-top: 23px; color: #77899d; font-size: 11px; line-height: 1.8; }.job-status > div { display: flex; gap: 11px; align-items: center; flex-wrap: wrap; }.job-status b { font-weight: 500; color: #4d6884; }.job-status > div > span:last-child { flex-basis: 100%; overflow-wrap: anywhere; }.job-status p { margin: 5px 0; }.published-snapshot { display: grid; grid-template-columns: 1.4fr 1fr; gap: 20px 30px; }.published-snapshot > div { display: flex; flex-direction: column; gap: 9px; }.published-snapshot span { color: var(--muted); font-size: 11px; }.published-snapshot b { color: #496078; font-size: 12px; font-weight: 500; overflow-wrap: anywhere; }.formal-log { background: #0e2133; color: #b6c7d6; border: 1px solid #163047; border-radius: 6px; padding: 20px; font-family: Consolas, 'Microsoft YaHei', monospace; font-size: 11px; line-height: 1.95; height: 325px; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; }
@media(max-width: 950px) { .formal-phases { grid-template-columns: 1fr 1fr; gap: 23px; } } @media(max-width: 650px) { .published-snapshot { grid-template-columns: 1fr; }.formal-log { padding: 15px; height: 280px; }.formal-phases { gap: 18px 10px; }.formal-phases li { gap: 9px; }.pipeline-actions > button { width: 100%; } }
</style>
