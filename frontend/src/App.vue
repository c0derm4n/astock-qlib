<template>
  <el-container class="layout">
    <el-aside width="200px">
      <div class="logo">A股ETF轮动系统</div>
      <el-menu :default-active="$route.path" router background-color="#001529" text-color="#a6adb4" active-text-color="#fff">
        <el-menu-item index="/">
          <el-icon><Odometer /></el-icon><span>总览</span>
        </el-menu-item>
        <el-menu-item index="/picks">
          <el-icon><Tickets /></el-icon><span>选ETF清单</span>
        </el-menu-item>
        <el-menu-item index="/decision">
          <el-icon><Bell /></el-icon><span>每日决策</span>
        </el-menu-item>
        <el-menu-item index="/backtest">
          <el-icon><TrendCharts /></el-icon><span>回测分析</span>
        </el-menu-item>
        <el-menu-item index="/pipeline">
          <el-icon><VideoPlay /></el-icon><span>数据管道</span>
        </el-menu-item>
      </el-menu>
    </el-aside>
    <el-container>
      <el-header class="header">
        <span class="page-title">{{ $route.meta.title }}</span>
        <el-tag v-if="pipe.running" type="warning" effect="dark">
          任务运行中：{{ stepName(pipe.current_step) }}
        </el-tag>
        <el-tag v-else-if="pipe.error" type="danger">上次任务失败</el-tag>
      </el-header>
      <el-main>
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { onMounted, onUnmounted, reactive } from 'vue'
import { Odometer, Tickets, TrendCharts, VideoPlay, Bell } from '@element-plus/icons-vue'
import api from './api'

const STEP_NAMES = { fetch: '抓取行情', dump: '转换数据', train: '训练回测', decide: '每日决策' }
const stepName = (s) => STEP_NAMES[s] || s || ''

const pipe = reactive({ running: false, current_step: null, error: null })
let timer = null

async function poll() {
  try {
    const s = await api.get('/pipeline/status')
    pipe.running = s.running
    pipe.current_step = s.current_step
    pipe.error = s.error
  } catch {
    /* 后端未启动时静默 */
  }
}

onMounted(() => {
  poll()
  timer = setInterval(poll, 5000)
})
onUnmounted(() => clearInterval(timer))
</script>

<style>
html,
body,
#app {
  height: 100%;
  margin: 0;
}
.layout {
  height: 100%;
}
.el-aside {
  background-color: #001529;
}
.logo {
  color: #fff;
  font-size: 16px;
  font-weight: 600;
  text-align: center;
  padding: 18px 0;
}
.el-menu {
  border-right: none;
}
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #e4e7ed;
  background: #fff;
}
.page-title {
  font-size: 16px;
  font-weight: 600;
}
.el-main {
  background: #f0f2f5;
}
</style>
