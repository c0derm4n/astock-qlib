<template>
  <section class="weekly-allocation" :aria-labelledby="`${strategy.id}-allocation-title`">
    <div class="allocation-heading">
      <div><h3 :id="`${strategy.id}-allocation-title`">仓位占比 <span>最近{{ weeks.length || 50 }}周</span></h3><p>{{ isDividendBand ? '周首信号日收盘等值持仓 / 账户总资产' : '每周收盘持仓市值 / 账户总资产' }} · 柱高为总投资仓位</p></div>
      <div class="allocation-legend" aria-label="仓位颜色图例"><span v-for="asset in assets" :key="asset.code"><i :style="{ background: asset.color }"></i>{{ asset.name }}</span></div>
    </div>
    <template v-if="weeks.length">
      <div ref="plot" class="allocation-plot" role="img" :aria-label="chartDescription"></div>
      <p class="allocation-note">从左到右由旧到新；空仓周保留为0%。分色部分为{{ isDividendBand ? '红利低波与国债的等值仓位' : '各ETF仓位' }}，柱顶间隔标注总仓位；拖动底部滑块可放大区间，悬停或展开明细可查看每周精确占比。{{ isDividendBand ? '按最近50个周首信号日的收盘账本展示。' : `本周按截至${strategy.data_end}的收盘持仓展示，未完成周标记＊。` }}现金与分红应收不计入投资仓位。</p>
      <details class="allocation-data"><summary>查看每周占比</summary><div class="allocation-table-scroll"><table><caption class="visually-hidden">最近{{ weeks.length }}周投资仓位，按账户总资产计算</caption><thead><tr><th>周首 / 估值日</th><th v-for="asset in assets" :key="asset.code">{{ asset.name }}</th><th>总投资仓位</th></tr></thead><tbody><tr v-for="week in weeks" :key="week.week_start"><td>{{ week.week_start }}{{ week.partial ? '＊' : '' }}<small>{{ week.carried_over ? '休市沿用 ' : '估值 ' }}{{ week.date }}</small></td><td v-for="asset in assets" :key="asset.code">{{ pct(weight(week, asset.code)) }}</td><td><b>{{ pct(totalWeight(week)) }}</b></td></tr></tbody></table></div></details>
    </template>
    <p v-else class="allocation-empty">暂无最近50周持仓数据，请更新正式策略快照。</p>
  </section>
</template>

<script setup>
import { computed, ref, watch, onMounted, onBeforeUnmount, nextTick } from 'vue'
import * as echarts from 'echarts/core'
import { BarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, MarkPointComponent, DataZoomComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { isDividendBandStrategy, strategyDisplayName } from '../strategyPresentation.js'
import { allocationWeeks, allocationAssets, allocationWeight, totalAllocationWeight } from '../weeklyAllocation.js'

echarts.use([BarChart, GridComponent, TooltipComponent, MarkPointComponent, DataZoomComponent, CanvasRenderer])
const props = defineProps({ strategy: { type: Object, required: true } })
const plot = ref(null)
const isDividendBand = computed(() => isDividendBandStrategy(props.strategy))
const weeks = computed(() => allocationWeeks(props.strategy))
const weight = allocationWeight
const totalWeight = totalAllocationWeight
const pct = value => value == null || !Number.isFinite(Number(value)) ? '—' : (value * 100).toFixed(2) + '%'
const assets = computed(() => allocationAssets(props.strategy, weeks.value))
const chartDescription = computed(() => `${strategyDisplayName(props.strategy)}最近${weeks.value.length}周仓位堆叠柱状图。` + weeks.value.map(week => `${week.week_start}：${assets.value.map(asset => `${asset.name}${pct(weight(week, asset.code))}`).join('，')}，总投资仓位${pct(totalWeight(week))}`).join('；'))
let chart, observer
let zoom = { start: 0, end: 100 }
function draw() {
  if (!plot.value) { observer?.disconnect(); chart?.dispose(); chart = null; return }
  if (!chart) { chart = echarts.init(plot.value); chart.on('datazoom', event => { const change = event.batch?.[0] || event; zoom = { start: change.start ?? zoom.start, end: change.end ?? zoom.end } }); observer = new ResizeObserver(() => { chart?.resize(); draw() }); observer.observe(plot.value) }
  const rows = weeks.value, seriesAssets = assets.value
  const narrow = plot.value.clientWidth < 480
  const visibleWeeks = Math.max(1, rows.length * (zoom.end - zoom.start) / 100)
  const pixelsPerWeek = (plot.value.clientWidth - 63) / visibleWeeks
  const labelStride = Math.max(1, Math.ceil(visibleWeeks / Math.max(1, Math.floor((plot.value.clientWidth - 63) / (narrow ? 26 : 42)))))
  chart.setOption({
    animation: false,
    textStyle: { fontFamily: 'Microsoft YaHei, sans-serif', color: '#727d89' },
    grid: { left: 48, right: 15, top: 28, bottom: 78 },
    tooltip: { trigger: 'axis', renderMode: 'richText', confine: true, axisPointer: { type: 'shadow' }, backgroundColor: '#fff', borderColor: '#dce2e8', textStyle: { color: '#25364a', fontSize: 12 },
      formatter: params => {
        const week = rows[params[0]?.dataIndex]
        if (!week) return ''
        return [`${week.week_start} — ${week.week_end}`, `${week.partial ? '本周截至' : week.carried_over ? '整周休市，沿用' : '收盘估值'} ${week.date}`, `总投资仓位  ${pct(totalWeight(week))}`,
          ...seriesAssets.map(asset => `${asset.name}  ${pct(weight(week, asset.code))}`), `可用现金  ${pct(week.cash_weight)}`,
          ...(week.receivable_weight > 0 ? [`分红应收  ${pct(week.receivable_weight)}`] : [])].join('\n')
      } },
    xAxis: { type: 'category', data: rows.map(week => week.week_start), axisTick: { show: false }, axisLine: { lineStyle: { color: '#dce2e8' } }, axisLabel: { interval: 'auto', hideOverlap: true, fontSize: 10, rotate: 35, formatter: (value, index) => value.slice(2) + (rows[index].partial ? '＊' : '') } },
    yAxis: { type: 'value', min: 0, max: 100, interval: 20, axisLabel: { formatter: '{value}%', fontSize: 11 }, splitLine: { lineStyle: { color: '#ecf0f3', type: 'dashed' } } },
    dataZoom: [{ type: 'inside', start: zoom.start, end: zoom.end, zoomOnMouseWheel: 'ctrl' }, { type: 'slider', start: zoom.start, end: zoom.end, height: 18, bottom: 5, borderColor: 'transparent', fillerColor: '#e5eaf2', handleStyle: { color: '#8395a9' }, showDetail: false }],
    series: seriesAssets.map((asset, index) => ({ name: asset.name, type: 'bar', stack: 'investment', barMaxWidth: 44, itemStyle: { color: asset.color }, emphasis: { focus: 'series' },
      data: rows.map(week => {
        const value = weight(week, asset.code) * 100
        return { value, label: { show: pixelsPerWeek >= 40 && value >= 8, position: 'inside', color: '#fff', fontSize: 11, formatter: value.toFixed(1) + '%' } }
      }),
      markPoint: index === seriesAssets.length - 1 ? { symbol: 'circle', symbolSize: 1, silent: true, itemStyle: { color: 'transparent' }, label: { show: true, position: 'top', color: '#42566e', fontSize: narrow ? 9 : 11, formatter: item => Number(item.value).toFixed(narrow ? 0 : 1) + '%' }, data: rows.map((week, index) => ({ coord: [week.week_start, totalWeight(week) * 100], value: totalWeight(week) * 100, label: { show: index === rows.length - 1 || (index % labelStride === 0 && index <= rows.length - 1 - labelStride) } })) } : undefined,
    })),
  }, true)
}
watch(() => props.strategy, async () => { zoom = { start: 0, end: 100 }; await nextTick(); draw() })
onMounted(draw)
onBeforeUnmount(() => { observer?.disconnect(); chart?.dispose() })
</script>

<style scoped>
.weekly-allocation { margin-top: 28px; padding-top: 25px; border-top: 1px solid var(--line); }
.allocation-heading { display: flex; align-items: center; justify-content: space-between; gap: 15px; }
h3 { margin: 0 0 8px; font-size: 17px; font-weight: 600; } h3 span { color: var(--muted); font-size: 11px; font-weight: 400; margin-left: 10px; }
.allocation-heading p, .allocation-note, .allocation-empty { color: var(--muted); font-size: 11px; line-height: 1.9; margin: 0; }
.allocation-legend { display: flex; flex-wrap: wrap; gap: 15px; font-size: 11px; color: #5f7186; }
.allocation-legend span { display: inline-flex; align-items: center; gap: 7px; }.allocation-legend i { width: 9px; height: 9px; border-radius: 2px; }
.allocation-plot { width: 100%; height: 300px; margin-top: 18px; }.allocation-note { margin-top: 5px; }.allocation-empty { padding: 24px 0; }
.allocation-data { margin-top: 14px; font-size: 11px; color: #6b7e91; }.allocation-data summary { cursor: pointer; padding: 4px 0; }.allocation-data summary:focus-visible { outline: 2px solid #5a8fdb; outline-offset: 3px; }
.allocation-table-scroll { overflow-x: auto; margin-top: 12px; } table { border-collapse: collapse; width: 100%; min-width: 440px; font-size: 11px; } th, td { padding: 10px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; } th { color: #7d8995; background: #f7f9fa; font-weight: 500; } th:first-child, td:first-child { text-align: left; } td small { display: block; font-size: 10px; color: var(--muted); margin-top: 4px; } td b { color: #42566e; font-weight: 500; }
.visually-hidden { position: absolute; width: 1px; height: 1px; padding: 0; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; }
@media(max-width: 680px) { .allocation-heading { align-items: flex-start; flex-direction: column; gap: 13px; }.allocation-plot { height: 265px; } }
</style>
