import { createRouter, createWebHashHistory } from 'vue-router'

// 哈希路由：生产模式由 FastAPI 直接托管静态文件，无需服务端重写规则
const routes = [
  { path: '/', name: 'strategies', component: () => import('./views/StrategiesView.vue'), meta: { title: 'ETF 策略工作台' } },
  { path: '/development', name: 'dashboard', component: () => import('./views/DashboardView.vue'), meta: { title: '正式策略研究总览' } },
  { path: '/development/picks', name: 'picks', component: () => import('./views/PicksView.vue'), meta: { title: '选ETF清单' } },
  { path: '/development/decision', name: 'decision', component: () => import('./views/DecisionView.vue'), meta: { title: '周度调仓' } },
  { path: '/development/backtest', name: 'backtest', component: () => import('./views/BacktestView.vue'), meta: { title: '回测分析' } },
  { path: '/development/pipeline', name: 'pipeline', component: () => import('./views/PipelineView.vue'), meta: { title: '数据管道' } },
  { path: '/legacy', redirect: '/development' },
  { path: '/picks', redirect: '/development/picks' },
  { path: '/decision', redirect: '/development/decision' },
  { path: '/backtest', redirect: '/development/backtest' },
  { path: '/pipeline', redirect: '/development/pipeline' },
]

export default createRouter({
  history: createWebHashHistory(),
  routes,
})
