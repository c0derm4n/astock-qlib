<template>
  <router-view v-if="$route.name === 'strategies'" />
  <div v-else class="development-shell">
    <aside class="development-sidebar">
      <router-link to="/development" class="development-brand"><span>Q</span><div><b>开发后台</b><small>WEEKLY RESEARCH</small></div></router-link>
      <nav aria-label="开发后台功能">
        <router-link v-for="item in navigation" :key="item.path" :to="item.path"><el-icon><component :is="item.icon" /></el-icon><span>{{ item.label }}</span></router-link>
      </nav>
      <div class="sidebar-footer"><router-link to="/">← 返回策略工作台</router-link><p>稳健 / 平衡 / 积极</p><span>正式周频 · 本地服务</span></div>
    </aside>
    <div class="development-workspace">
      <header class="development-header"><div><span>开发后台</span><b>{{ $route.meta.title }}</b></div><span v-if="pipe.running" class="development-job running">正式策略刷新运行中</span><span v-else-if="pipe.error" class="development-job failed">最近刷新失败</span><span v-else class="development-job ready">正式策略服务</span></header>
      <main class="development-content"><router-view /></main>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, reactive, watch } from 'vue'
import { useRoute } from 'vue-router'
import { Odometer, Tickets, TrendCharts, VideoPlay, Bell } from '@element-plus/icons-vue'
import api from './api'
const route = useRoute()
const navigation = [
  { path: '/development', label: '正式策略研究总览', icon: Odometer },
  { path: '/development/picks', label: '选ETF清单', icon: Tickets },
  { path: '/development/decision', label: '周度调仓', icon: Bell },
  { path: '/development/backtest', label: '回测分析', icon: TrendCharts },
  { path: '/development/pipeline', label: '数据管道', icon: VideoPlay },
]
const pipe = reactive({ running: false, error: null })
let timer
async function poll() {
  if (route.name === 'strategies') return
  try { const data = await api.get('/strategies/refresh/status', { silentErrors: true }); pipe.running = data.running; pipe.error = data.error }
  catch { /* 页面自身呈现连接错误 */ }
}
watch(() => route.name, poll)
onMounted(() => { poll(); timer = setInterval(poll, 4000) })
onUnmounted(() => clearInterval(timer))
</script>

<style>
html, body, #app { min-height: 100%; margin: 0; }
.development-shell { --line: #e2e7ec; --muted: #7c8795; min-height: 100vh; color: #25364a; font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; display: flex; background: #f4f6f8; -webkit-font-smoothing: antialiased; }
.development-shell * { box-sizing: border-box; }
.development-sidebar { width: 220px; flex-shrink: 0; background: #13263b; color: #ccd6e0; padding: 29px 16px 24px; position: sticky; top: 0; height: 100vh; display: flex; flex-direction: column; }
.development-brand { display: flex; gap: 12px; align-items: center; text-decoration: none; color: #fff; padding: 0 9px 31px; }
.development-brand > span { display: grid; place-items: center; height: 34px; width: 34px; border: 1px solid #6e829950; border-radius: 7px; font: 25px Georgia, serif; }
.development-brand b { font-size: 17px; font-weight: 600; }.development-brand small { display: block; font-size: 8px; letter-spacing: 1.9px; color: #8297ad; margin-top: 6px; }
.development-sidebar nav { display: flex; flex-direction: column; gap: 5px; }.development-sidebar nav a { padding: 13px 12px; font-size: 13px; color: #aab9c9; text-decoration: none; display: flex; gap: 12px; align-items: center; border-radius: 5px; transition: background .15s; }.development-sidebar nav a:hover { background: #203952; color: #fff; }.development-sidebar nav a.router-link-exact-active { color: #fff; background: #2c4c70; }.development-sidebar nav .el-icon { font-size: 16px; }
.sidebar-footer { margin-top: auto; border-top: 1px solid #31475c; padding: 23px 10px 0; font-size: 11px; color: #8fa2b5; }.sidebar-footer a { color: #cbd9e6; text-decoration: none; font-size: 12px; }.sidebar-footer p { margin: 20px 0 7px; }.sidebar-footer span { font-size: 10px; color: #6f859b; }
.development-workspace { flex: 1; min-width: 0; }.development-header { height: 72px; padding: 0 32px; background: #fff; border-bottom: 1px solid var(--line); display: flex; align-items: center; justify-content: space-between; gap: 15px; }.development-header > div { display: flex; gap: 18px; align-items: center; }.development-header span { font-size: 11px; color: #8695a5; }.development-header b { font-size: 15px; font-weight: 500; border-left: 1px solid var(--line); padding-left: 18px; }.development-header .development-job { padding: 6px 9px; border-radius: 4px; font-size: 10px; background: #eef3f8; color: #5d7894; }.development-header .running { color: #9a7c46; background: #faf6ed; }.development-header .failed { color: #a45e51; background: #fbefec; }
.development-content { padding: 30px 32px 35px; max-width: 1500px; margin: 0 auto; }.development-shell button, .development-shell input, .development-shell select { font-family: inherit; }.development-shell a:focus-visible, .development-shell button:focus-visible { outline: 2px solid #78a4e2; outline-offset: 3px; }
@media(max-width: 1050px) { .development-sidebar { width: 195px; padding-inline: 12px; }.development-header { padding-inline: 24px; }.development-content { padding: 24px; } }
@media(max-width: 720px) { .development-shell { display: block; }.development-sidebar { width: 100%; height: auto; position: static; padding: 16px; }.development-brand { padding: 0 4px 17px; }.development-sidebar nav { flex-direction: row; flex-wrap: wrap; gap: 5px; }.development-sidebar nav a { padding: 9px 10px; font-size: 11px; gap: 7px; }.sidebar-footer { margin-top: 16px; padding-top: 14px; border-top-color: #304359; }.sidebar-footer p, .sidebar-footer span { display: none; }.development-header { height: 61px; padding-inline: 16px; }.development-header > div { gap: 10px; }.development-header b { font-size: 13px; padding-left: 10px; }.development-header .development-job { font-size: 9px; }.development-content { padding: 20px 12px; } }
@media(prefers-reduced-motion: reduce) { .development-shell * { transition: none !important; } }
</style>
