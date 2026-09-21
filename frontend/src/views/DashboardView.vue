<template>
  <div v-loading="loading">
    <el-alert v-if="acceptance" class="mb16" :type="acceptance.gate.approved ? 'success' : 'warning'" :closable="false"
      :title="acceptance.gate.approved ? '研究验收满足预设门槛，仍需人工投资审核' : '投资准入未通过：当前仅限模拟研究'"
      :description="`前瞻观察 ${acceptance.gate.forward_days} 个交易日；未通过项：${gateFailures(acceptance.gate.failed_checks)}`" />
    <el-empty v-if="!metrics && !picks.length" description="暂无数据：请先到「数据管道」运行 训练(train)" />
    <template v-else>
      <!-- 预测质量 -->
      <el-card v-if="metrics" class="mb16">
        <template #header>
          <span>预测质量（滚动历史研究）</span>
          <span class="card-sub">训练时间：{{ metrics.finished_at || '-' }}<template v-if="metrics.data_version">　数据版本：{{ metrics.data_version }}</template></span>
        </template>
        <el-row :gutter="16">
          <el-col :span="6"><el-statistic title="IC均值" :value="fmt(metrics.ic?.['IC均值'])" /></el-col>
          <el-col :span="6"><el-statistic title="ICIR" :value="fmt(metrics.ic?.['ICIR'])" /></el-col>
          <el-col :span="6"><el-statistic title="RankIC均值" :value="fmt(metrics.ic?.['RankIC均值'])" /></el-col>
          <el-col :span="6"><el-statistic title="RankICIR" :value="fmt(metrics.ic?.['RankICIR'])" /></el-col>
        </el-row>
        <div class="hint">相关系数应结合时间相关性校正后的置信区间判断，单一阈值不能证明策略有效。</div>
      </el-card>

      <!-- 回测概览 -->
      <el-card v-if="bt && Object.keys(bt).length" class="mb16">
        <template #header>
          <span>回测概览（固定研究协议）</span>
          <span class="card-sub">区间：{{ bt.start }} ~ {{ bt.end }}（轮动最多 {{ bt.topk }} 只ETF，底仓另计<template v-if="bt.slippage_bps != null">，滑点 {{ bt.slippage_bps }}bp</template>）</span>
        </template>
        <el-row :gutter="16">
          <el-col :span="4">
            <el-statistic title="策略累计收益(扣费)" :value="pct(bt.strategy_cum_return_pct)" />
          </el-col>
          <el-col :span="4">
            <el-statistic title="沪深300ETF参考" :value="pct(bt.hs300_cum_return_pct)" />
          </el-col>
          <el-col :span="4">
            <el-statistic title="等权ETF参考" :value="pct(bt.eqw_cum_return_pct)" />
          </el-col>
          <el-col :span="4">
            <el-statistic title="策略复合年化" :value="pct(bt.strategy_annual_pct)" />
          </el-col>
          <el-col :span="4"><el-statistic title="策略最大回撤" :value="pct(bt.strategy_max_drawdown_pct)" /></el-col>
          <el-col :span="4">
            <el-statistic title="股债金参考累计" :value="pct(acceptance?.summaries?.balanced?.cumulative_return == null ? null : acceptance.summaries.balanced.cumulative_return * 100)" />
          </el-col>
        </el-row>
        <div class="hint" v-if="bt.strategy_max_drawdown_pct != null">
          对照策略均计入成本与建仓过程；沪深300ETF参考采用95%仓位上限，其余保留现金。
        </div>
      </el-card>

      <!-- 最新选 ETF -->
      <el-card v-if="picks.length" class="mb16">
        <template #header>
          <span>最新选ETF（{{ picksDate }}）</span>
          <el-link type="primary" class="card-sub" @click="$router.push('/picks')">查看历史清单 →</el-link>
        </template>
        <el-table :data="picks" size="small" style="width: 100%">
          <el-table-column prop="rank" label="排名" width="80" />
          <el-table-column prop="code" label="代码" width="120" />
          <el-table-column prop="name" label="名称" width="160" />
          <el-table-column prop="score" label="打分">
            <template #default="{ row }">{{ row.score.toFixed(4) }}</template>
          </el-table-column>
        </el-table>
      </el-card>

      <el-alert type="warning" :closable="false" title="模型打分排名仅作决策支持，不是投资建议；请结合基本面与风控，先模拟盘验证。" />
    </template>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import api from '../api'

const loading = ref(true)
const metrics = ref(null)
const bt = ref({})
const picks = ref([])
const picksDate = ref('')
const acceptance = ref(null)
const gateLabels = {
  historical_net_positive: '历史扣费收益', drawdown_within_budget: '回撤预算', majority_full_years_positive: '多数年度盈利',
  double_slippage_positive: '双倍滑点', excess_vs_momentum_confident: '超越简单动量的统计证据',
  execution_stress_survives: '执行压力下收益与回撤',
  forward_observation_sufficient: '前瞻观察时长', historical_universe_audited: '历史标的完整性审计',
  corporate_actions_audited: '分红与份额变动审计', forward_performance_acceptable: '前瞻表现与执行一致性',
  full_stress_executed: '完整压力测试', parameter_neighborhood_survives: '参数邻域稳健性', market_data_fresh: '行情时效性',
  market_coverage_complete: '行情标的覆盖完整性',
}
const gateFailures = (keys) => keys.map((key) => gateLabels[key] || key).join('、') || '无'

const fmt = (v) => (v === null || v === undefined ? '-' : Number(v).toFixed(4))
const pct = (v) => (v === null || v === undefined ? '-' : `${Number(v).toFixed(2)}%`)

onMounted(async () => {
  try {
    const [m, p, a] = await Promise.allSettled([api.get('/train/metrics'), api.get('/picks'), api.get('/acceptance', { silentErrors: true })])
    if (a.status === 'fulfilled') acceptance.value = a.value
    if (m.status === 'fulfilled') {
      metrics.value = m.value
      bt.value = m.value.backtest || {}
    }
    if (p.status === 'fulfilled') {
      picks.value = p.value.rows
      picksDate.value = p.value.date
    }
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.mb16 {
  margin-bottom: 16px;
}
.card-sub {
  float: right;
  font-size: 13px;
  color: #909399;
  font-weight: normal;
}
.hint {
  margin-top: 8px;
  font-size: 12px;
  color: #909399;
}
</style>
