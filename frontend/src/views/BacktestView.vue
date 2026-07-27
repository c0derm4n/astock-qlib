<template>
  <div v-loading="loading">
    <template v-if="summary">
      <el-card class="mb16">
        <template #header>回测汇总（{{ summary.start }} ~ {{ summary.end }}，共 {{ summary.days }} 个交易日）</template>
        <el-descriptions :column="4" border size="small">
          <el-descriptions-item label="策略累计收益(扣费)">{{ pct(summary.strategy_cum_return) }}</el-descriptions-item>
          <el-descriptions-item label="策略年化">{{ pct(summary.strategy_annual) }}</el-descriptions-item>
          <el-descriptions-item label="沪深300累计(代理)">{{ pct(summary.bench_cum_return) }}</el-descriptions-item>
          <el-descriptions-item label="超额年化(vs沪深300)">{{ pct(summary.excess_annual) }}</el-descriptions-item>
          <el-descriptions-item label="信息比率IR">{{ fmt(summary.information_ratio) }}</el-descriptions-item>
          <el-descriptions-item label="最大回撤">{{ pct(summary.max_drawdown) }}</el-descriptions-item>
        </el-descriptions>
      </el-card>

      <el-card>
        <template #header>累计收益曲线（策略扣费后 vs 沪深300）</template>
        <div ref="chartEl" class="chart"></div>
      </el-card>
    </template>
    <el-empty v-else-if="!loading" description="暂无回测报告：请先到「数据管道」运行 训练(train)" />
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref } from 'vue'
import * as echarts from 'echarts'
import api from '../api'

const loading = ref(true)
const summary = ref(null)
const chartEl = ref(null)
let chart = null

const pct = (v) => (v === null || v === undefined ? '-' : `${(v * 100).toFixed(2)}%`)
const fmt = (v) => (v === null || v === undefined ? '-' : v)

function render(data) {
  if (!chartEl.value) return
  chart = echarts.init(chartEl.value)
  chart.setOption({
    tooltip: {
      trigger: 'axis',
      valueFormatter: (v) => `${(v * 100).toFixed(2)}%`,
    },
    legend: { data: ['策略(扣费)', '沪深300'] },
    grid: { left: 60, right: 30, top: 40, bottom: 80 },
    xAxis: { type: 'category', data: data.dates },
    yAxis: { type: 'value', axisLabel: { formatter: (v) => `${(v * 100).toFixed(0)}%` } },
    dataZoom: [{ type: 'inside' }, { type: 'slider' }],
    series: [
      { name: '策略(扣费)', type: 'line', data: data.strategy, showSymbol: false },
      { name: '沪深300', type: 'line', data: data.benchmark, showSymbol: false },
    ],
  })
}

function onResize() {
  chart?.resize()
}

onMounted(async () => {
  try {
    const [s, r] = await Promise.all([api.get('/backtest/summary'), api.get('/backtest/report')])
    summary.value = s
    // 等 DOM 渲染出 chartEl 再初始化
    setTimeout(() => render(r), 0)
  } catch {
    /* 无回测数据时显示空态 */
  } finally {
    loading.value = false
  }
  window.addEventListener('resize', onResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', onResize)
  chart?.dispose()
})
</script>

<style scoped>
.mb16 {
  margin-bottom: 16px;
}
.chart {
  width: 100%;
  height: 420px;
}
</style>
