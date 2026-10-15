<template>
  <div class="chart-frame">
    <div class="chart-topline">
      <div><h3>{{ title }}</h3><p>{{ subtitle }}</p></div>
      <div class="chart-tabs" role="group" aria-label="选择图表">
        <button v-for="tab in tabs" :key="tab.id" :class="{ active: kind === tab.id }" :aria-pressed="kind === tab.id" @click="kind = tab.id">{{ tab.label }}</button>
      </div>
    </div>
    <div ref="plot" class="strategy-plot" role="img" :aria-label="`${strategy.name} ${title}，区间 ${strategy.data_start} 至 ${strategy.data_end}`"></div>
    <p class="chart-footnote">{{ kind.startsWith('annual') ? '首尾不足全年单独标记；年度回撤包含上年末资金作为起始峰值。' : '每日收盘估值，扣除佣金、滑点与冲击成本；仅在周首信号触发后于下一交易日开盘调仓。' }}</p>
  </div>
</template>

<script setup>
import { computed, ref, watch, onMounted, onBeforeUnmount, nextTick } from 'vue'
import * as echarts from 'echarts/core'
import { LineChart, BarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, DataZoomComponent, MarkLineComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
echarts.use([LineChart, BarChart, GridComponent, TooltipComponent, DataZoomComponent, MarkLineComponent, CanvasRenderer])
const props = defineProps({ strategy: { type: Object, required: true } })
const tabs = [{ id: 'capital', label: '资金曲线' }, { id: 'annualReturn', label: '年度收益' }, { id: 'annualDrawdown', label: '年度回撤' }, { id: 'drawdown', label: '回撤曲线' }]
const kind = ref('capital'), plot = ref(null)
const title = computed(() => tabs.find(t => t.id === kind.value).label)
const subtitle = computed(() => kind.value === 'capital' ? '100万元起始 · 单位：万元' : kind.value === 'annualReturn' ? '各自然年的账户收益 · 单位：%' : '资金相对历史峰值的损失 · 单位：%')
let chart, observer
const n = (v, digits = 2) => Number(v).toLocaleString('zh-CN', { minimumFractionDigits: digits, maximumFractionDigits: digits })
function draw() {
  if (!chart) return
  const s = props.strategy, bar = kind.value.startsWith('annual')
  const annual = s.metrics.annual
  const dates = bar ? annual.map(a => `${a.year}${a.complete ? '' : '＊'}`) : s.series.dates
  const values = bar ? annual.map(a => (kind.value === 'annualReturn' ? a.return : a.max_drawdown) * 100) : (kind.value === 'capital' ? s.series.capital.map(v => v / 10000) : s.series.drawdown.map(v => v * 100))
  chart.setOption({
    animation: false, textStyle: { fontFamily: 'Microsoft YaHei, sans-serif', color: '#727d89' },
    grid: { left: 62, right: 24, top: 20, bottom: bar ? 42 : 68 },
    tooltip: { trigger: 'axis', backgroundColor: '#fff', borderColor: '#dce2e8', textStyle: { color: '#1c2a3a' },
      valueFormatter: v => n(v) + (kind.value === 'capital' ? ' 万元' : '%') },
    xAxis: { type: 'category', data: dates, boundaryGap: bar, axisLine: { lineStyle: { color: '#dce2e8' } }, axisTick: { show: false }, axisLabel: { hideOverlap: true, formatter: v => bar ? v : v.slice(0, 7) } },
    yAxis: { type: 'value', scale: kind.value === 'capital', axisLabel: { formatter: v => v + (kind.value === 'capital' ? '' : '%') }, splitLine: { lineStyle: { color: '#ecf0f3', type: 'dashed' } } },
    dataZoom: bar ? [] : [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 10, borderColor: 'transparent', fillerColor: '#e5eaf2', handleStyle: { color: s.color }, showDetail: false }],
    series: [{ name: s.name, type: bar ? 'bar' : 'line', data: values, showSymbol: false, barMaxWidth: 44,
      itemStyle: { color: params => kind.value === 'annualReturn' && params.value < 0 ? '#b45a52' : s.color, borderRadius: bar ? [3, 3, 0, 0] : 0 },
      lineStyle: { color: s.color, width: 2 }, areaStyle: bar ? undefined : { color: s.color, opacity: .065 },
      label: { show: bar, position: 'top', color: '#5c6672', fontSize: 11, formatter: p => n(p.value, 1) + '%' },
      markLine: kind.value === 'capital' ? { symbol: 'none', label: { formatter: '初始100万', position: 'insideEndTop' }, lineStyle: { color: '#aab4bf', type: 'dashed' }, data: [{ yAxis: 100 }] } : undefined }],
  }, true)
}
watch([kind, () => props.strategy], async () => { await nextTick(); draw() })
onMounted(() => { chart = echarts.init(plot.value); observer = new ResizeObserver(() => chart.resize()); observer.observe(plot.value); draw() })
onBeforeUnmount(() => { observer?.disconnect(); chart?.dispose() })
</script>

<style scoped>
.chart-frame { margin-top: 32px; border-top: 1px solid var(--line); padding-top: 25px; }
.chart-topline { display: flex; justify-content: space-between; gap: 15px; align-items: center; }
h3 { margin: 0 0 7px; font-size: 17px; font-weight: 600; } p { margin: 0; color: var(--muted); font-size: 12px; }
.chart-tabs { display: flex; gap: 4px; background: #f1f4f6; border-radius: 7px; padding: 4px; flex-wrap: wrap; }
button { border: 0; background: transparent; padding: 8px 12px; border-radius: 4px; color: #727d89; cursor: pointer; font-size: 12px; }
button.active { background: white; color: #20334a; box-shadow: 0 1px 4px #1a2c4212; } button:hover { color: #20334a; }
.strategy-plot { width: 100%; height: 310px; margin-top: 20px; }
.chart-footnote { margin-top: 6px; line-height: 1.8; }
@media (max-width: 680px) { .chart-topline { align-items: flex-start; flex-direction: column; } .chart-tabs { width: 100%; } button { flex: 1; padding: 8px 6px; } .strategy-plot { height: 255px; } }
</style>
