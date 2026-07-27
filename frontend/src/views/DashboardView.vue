<template>
  <div v-loading="loading">
    <el-empty v-if="!metrics && !picks.length" description="暂无数据：请先到「数据管道」运行 训练(train)" />
    <template v-else>
      <!-- 预测质量 -->
      <el-card v-if="metrics" class="mb16">
        <template #header>
          <span>预测质量（测试集）</span>
          <span class="card-sub">训练时间：{{ metrics.finished_at || '-' }}</span>
        </template>
        <el-row :gutter="16">
          <el-col :span="6"><el-statistic title="IC均值" :value="fmt(metrics.ic?.['IC均值'])" /></el-col>
          <el-col :span="6"><el-statistic title="ICIR" :value="fmt(metrics.ic?.['ICIR'])" /></el-col>
          <el-col :span="6"><el-statistic title="RankIC均值" :value="fmt(metrics.ic?.['RankIC均值'])" /></el-col>
          <el-col :span="6"><el-statistic title="RankICIR" :value="fmt(metrics.ic?.['RankICIR'])" /></el-col>
        </el-row>
        <div class="hint">参考：RankIC均值 &gt; 0.03 有一定轮动力，&gt; 0.05 较好；RankICIR 越高越稳定</div>
      </el-card>

      <!-- 回测概览 -->
      <el-card v-if="bt && Object.keys(bt).length" class="mb16">
        <template #header>
          <span>回测概览</span>
          <span class="card-sub">区间：{{ bt.start }} ~ {{ bt.end }}（持仓 {{ bt.topk }} 只ETF）</span>
        </template>
        <el-row :gutter="16">
          <el-col :span="4">
            <el-statistic title="策略累计收益(扣费)"><template #default>{{ pct(bt.strategy_cum_return_pct) }}</template></el-statistic>
          </el-col>
          <el-col :span="4">
            <el-statistic title="沪深300累计"><template #default>{{ pct(bt.hs300_cum_return_pct) }}</template></el-statistic>
          </el-col>
          <el-col :span="4">
            <el-statistic title="等权ETF基准累计"><template #default>{{ pct(bt.eqw_cum_return_pct) }}</template></el-statistic>
          </el-col>
          <el-col :span="4">
            <el-statistic title="超额年化(vs沪深300)"><template #default>{{ pct(bt.excess_annual_pct) }}</template></el-statistic>
          </el-col>
          <el-col :span="4"><el-statistic title="信息比率IR" :value="fmt(bt.information_ratio)" /></el-col>
          <el-col :span="4">
            <el-statistic title="超额最大回撤"><template #default>{{ pct(bt.excess_max_drawdown_pct) }}</template></el-statistic>
          </el-col>
        </el-row>
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

const fmt = (v) => (v === null || v === undefined ? '-' : Number(v).toFixed(4))
const pct = (v) => (v === null || v === undefined ? '-' : `${Number(v).toFixed(2)}%`)

onMounted(async () => {
  try {
    const [m, p] = await Promise.allSettled([api.get('/train/metrics'), api.get('/picks')])
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
