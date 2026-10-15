<template>
  <div class="strategy-app">
    <header class="desk-header"><a class="desk-brand" href="#/"><span class="brand-mark">Q</span><b>ETF 策略工作台</b><span class="local-label">LOCAL</span></a><nav aria-label="策略导航"><a href="#lightgbm" @click.prevent="jump('lightgbm')">LightGBM</a><a href="#trend200_vol15" @click.prevent="jump('trend200_vol15')">通信 / 黄金</a><router-link to="/development">开发后台</router-link></nav><span class="connection"><i :class="{ offline: error }"></i>{{ error ? '服务待连接' : '本地服务' }}</span></header>
    <main class="desk-main">
      <div class="page-intro"><div><div class="page-eyebrow">WEEKLY STRATEGY DESK</div><h1>两条路线，一个周度节奏。</h1><p>从2016年9月12日开始，各自100万元初始资金。每周首个交易日收盘评估，触发才调仓。</p></div><button class="refresh-button" :disabled="job.running || refreshing" @click="refresh"><span class="refresh-icon" :class="{ spinning: job.running }">↻</span>{{ job.running ? '正在更新策略' : '更新行情与建议' }}</button></div>
      <div v-if="snapshot" class="snapshot-bar"><span><i class="data-dot"></i>行情截至 <b>{{ snapshot.data_end }}</b></span><span>快照生成 {{ formatTime(snapshot.completed_at) }}</span><span>收盘信号 · 次交易日开盘 · 100份整数买入</span></div>
      <div class="notice"><span class="notice-symbol">!</span><p><b>正式服务已登记，投资准入尚未通过。</b> 两条策略的历史年度亏损与年化收益尚未满足全部目标。页面提供模型账户和账户试算建议；实际账户请先核对持仓与现金。</p></div>
      <div v-if="snapshot?.refresh_error || job.error" class="error-notice" role="alert">上次刷新遇到问题：{{ job.error || snapshot.refresh_error }}。继续展示最后有效快照。</div>
      <div v-if="job.running" class="refresh-progress" role="status"><span class="progress-indicator"></span><div><b>行情校验 → 年度模型 → 周频回放 → 发布建议</b><p>{{ job.log_tail?.at(-1) || '正在启动…' }}</p></div></div>
      <div v-if="loading && !snapshot" class="loading-state" role="status"><span class="progress-indicator"></span><p>正在读取正式策略快照…</p></div>
      <div v-else-if="error" class="load-error" role="alert"><h2>暂时无法读取策略</h2><p>{{ error }}</p><button @click="load">重新连接</button><button @click="refresh" :disabled="job.running">构建正式快照</button></div>
      <template v-if="snapshot"><StrategySection v-for="(s, i) in snapshot.strategies" :key="s.id" :strategy="s" :index="i" :universe="universe" /></template>
      <footer class="desk-footer"><span>ETF 策略工作台 / v3.0.1</span><p>历史回放使用已查看的市场与候选，不构成策略选择独立盲测；两个策略的账户估值口径分别标注。现金利息为0。</p><span>Asia / Shanghai</span></footer>
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'
import api from '../api'
import StrategySection from '../components/StrategySection.vue'
const snapshot = ref(null), universe = ref([]), loading = ref(true), error = ref(''), refreshing = ref(false), job = ref({ running: false })
let timer, wasRunning = false
const formatTime = d => new Date(d).toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', hour12: false })
const jump = id => document.getElementById(id)?.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' })
async function load() {
  loading.value = true; error.value = ''
  try { snapshot.value = await api.get('/strategies', { silentErrors: true }); const pool = await api.get('/development/universe', { params: { strategy_id: 'lightgbm', run_id: snapshot.value.run_id }, silentErrors: true }); universe.value = pool.rows }
  catch (e) { error.value = e.response?.data?.detail || '请确认后台服务已启动，再重新连接。' }
  finally { loading.value = false }
}
async function poll() {
  try { job.value = await api.get('/strategies/refresh/status', { silentErrors: true }); if (wasRunning && !job.value.running) await load(); wasRunning = job.value.running } catch { /* 错误由主动加载呈现 */ }
}
async function refresh() {
  refreshing.value = true
  try { job.value = await api.post('/strategies/refresh', { fetch: true }); wasRunning = true }
  finally { refreshing.value = false }
}
onMounted(() => { load(); poll(); timer = setInterval(poll, 3000) })
onBeforeUnmount(() => clearInterval(timer))
</script>

<style>
.strategy-app { --line: #e2e7ec; --muted: #7c8795; min-height: 100vh; color: #25364a; background: #f4f6f8; font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; -webkit-font-smoothing: antialiased; } .strategy-app * { box-sizing: border-box; }.strategy-app button, .strategy-app input, .strategy-app select { font-family: inherit; }.strategy-app button:focus-visible, .strategy-app a:focus-visible, .strategy-app summary:focus-visible, .strategy-app input:focus-visible, .strategy-app select:focus-visible { outline: 2px solid #5a8fdb; outline-offset: 3px; }
.desk-header { height: 66px; position: sticky; top: 0; z-index: 20; background: #fffffff5; border-bottom: 1px solid var(--line); display: flex; align-items: center; padding: 0 5vw; justify-content: space-between; gap: 30px; backdrop-filter: blur(8px); }.desk-brand { display: flex; align-items: center; gap: 12px; color: #25364a; text-decoration: none; font-size: 14px; }.brand-mark { display: grid; place-items: center; width: 30px; height: 30px; border-radius: 6px; background: #233750; color: #fff; font-family: Georgia, serif; font-size: 22px; }.desk-brand b { font-weight: 600; }.local-label { color: #8c99a7; font-size: 8px; letter-spacing: 1.5px; border: 1px solid #e1e5e9; padding: 3px 4px; }.desk-header nav { display: flex; gap: 28px; margin-left: auto; }.desk-header nav a { color: #6e7e90; text-decoration: none; font-size: 12px; }.desk-header nav a:hover { color: #2b6ad6; }.connection { display: flex; gap: 6px; align-items: center; font-size: 10px; color: #8b97a5; padding-left: 20px; border-left: 1px solid var(--line); }.connection i, .data-dot { width: 5px; height: 5px; background: #78a991; border-radius: 50%; display: inline-block; }.connection i.offline { background: #bc6b62; }
.desk-main { max-width: 1240px; margin: 0 auto; padding: 44px 34px 22px; }.page-intro { display: flex; justify-content: space-between; align-items: center; gap: 20px; }.page-eyebrow { font-size: 10px; color: #738294; letter-spacing: 2.4px; } .page-intro h1 { margin: 13px 0 12px; font-size: 30px; letter-spacing: -.5px; font-weight: 600; }.page-intro p { font-size: 12px; color: var(--muted); line-height: 1.9; margin: 0; }.refresh-button { color: #fff; background: #253a53; border: 1px solid #253a53; border-radius: 6px; padding: 11px 17px; font-size: 12px; white-space: nowrap; cursor: pointer; }.refresh-button:hover { background: #304d70; }.refresh-button:disabled { opacity: .65; cursor: progress; }.refresh-icon { display: inline-block; font-size: 17px; margin-right: 8px; vertical-align: -1px; }.spinning { animation: rotate 1.5s linear infinite; } @keyframes rotate { to { transform: rotate(360deg); } }
.snapshot-bar { display: flex; gap: 23px; align-items: center; font-size: 10px; color: #8692a0; padding: 22px 0 18px; flex-wrap: wrap; }.snapshot-bar b { font-weight: 500; color: #4c6076; }.data-dot { margin-right: 7px; }.snapshot-bar span + span { border-left: 1px solid #dce2e8; padding-left: 22px; }.notice { display: flex; align-items: center; gap: 14px; padding: 14px 18px; margin-bottom: 24px; background: #faf8f2; border: 1px solid #e8e1d1; border-radius: 6px; }.notice-symbol { color: #a88b53; border: 1px solid #d6c6a6; border-radius: 50%; font-size: 11px; width: 17px; height: 17px; text-align: center; flex-shrink: 0; }.notice p { margin: 0; font-size: 11px; color: #8d7b61; line-height: 1.9; }.notice b { font-weight: 600; }.error-notice { color: #a96654; border: 1px solid #efd8ce; background: #fcf3ed; padding: 16px; margin-bottom: 20px; border-radius: 6px; font-size: 12px; word-break: break-word; }
.refresh-progress, .loading-state { border: 1px solid var(--line); padding: 20px; background: white; border-radius: 8px; display: flex; align-items: center; gap: 14px; margin-bottom: 24px; font-size: 12px; }.progress-indicator { width: 16px; height: 16px; border: 2px solid #d4dfea; border-top-color: #3d6d9d; border-radius: 50%; animation: rotate 1.2s linear infinite; flex-shrink: 0; }.refresh-progress b { font-weight: 500; }.refresh-progress p { color: var(--muted); font-size: 11px; margin: 8px 0 0; word-break: break-word; }.load-error { background: white; padding: 30px; margin-bottom: 25px; border: 1px solid var(--line); border-radius: 8px; }.load-error h2 { font-size: 20px; }.load-error p { font-size: 12px; color: var(--muted); }.load-error button { margin-right: 12px; border: 1px solid var(--line); border-radius: 4px; padding: 9px 14px; background: #f5f8fb; cursor: pointer; }.desk-footer { display: flex; align-items: center; gap: 24px; justify-content: space-between; font-size: 9px; color: #9ba5af; }.desk-footer p { flex: 1; line-height: 1.8; }.desk-footer > span { white-space: nowrap; }
@media(max-width: 850px) { .desk-header { padding: 0 22px; gap: 18px; }.connection { display: none; }.desk-header nav { gap: 16px; }.desk-main { padding: 32px 22px 20px; }.page-intro h1 { font-size: 25px; }.snapshot-bar { gap: 12px; } }
@media(max-width: 560px) { .desk-header { height: auto; min-height: 60px; padding: 13px 16px; flex-wrap: wrap; gap: 12px; }.desk-brand { font-size: 13px; }.desk-header nav { gap: 19px; width: 100%; justify-content: space-between; }.desk-main { padding: 26px 12px 18px; }.page-intro { flex-direction: column; align-items: flex-start; gap: 18px; padding: 0 5px; }.page-intro h1 { font-size: 24px; }.snapshot-bar { padding: 18px 5px; }.snapshot-bar span + span { border: 0; padding: 0; }.snapshot-bar span:last-child { width: 100%; }.notice { padding: 12px; gap: 9px; align-items: flex-start; }.notice-symbol { margin-top: 3px; }.desk-footer { flex-wrap: wrap; gap: 5px; padding: 0 5px; }.desk-footer p { flex-basis: 100%; order: 2; } }
@media(prefers-reduced-motion: reduce) { .strategy-app *, .strategy-app *::before, .strategy-app *::after { animation: none !important; scroll-behavior: auto !important; } }
</style>
