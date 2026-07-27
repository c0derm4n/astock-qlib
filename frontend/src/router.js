import { createRouter, createWebHashHistory } from 'vue-router'

// 哈希路由：生产模式由 FastAPI 直接托管静态文件，无需服务端重写规则
const routes = [
  { path: '/', name: 'dashboard', component: () => import('./views/DashboardView.vue'), meta: { title: '总览' } },
  { path: '/picks', name: 'picks', component: () => import('./views/PicksView.vue'), meta: { title: '选ETF清单' } },
  { path: '/decision', name: 'decision', component: () => import('./views/DecisionView.vue'), meta: { title: '每日决策' } },
  { path: '/backtest', name: 'backtest', component: () => import('./views/BacktestView.vue'), meta: { title: '回测分析' } },
  { path: '/pipeline', name: 'pipeline', component: () => import('./views/PipelineView.vue'), meta: { title: '数据管道' } },
]

export default createRouter({
  history: createWebHashHistory(),
  routes,
})
