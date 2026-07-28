<template>
  <div v-loading="loading">
    <template v-if="decision">
      <el-alert
        class="mb16"
        type="info"
        :closable="false"
        :title="`${decision.date} 盘中决策清单（每日 14:30 到「数据管道」运行 每日决策 更新；QDII 不参与）`"
      />
      <el-card>
        <div v-if="positions" class="pos-line">当前持仓：{{ positions.positions.length ? positions.positions.join('、') : '空仓（按 TopK 一次建仓）' }}</div>
        <el-table :data="decision.rows" stripe style="width: 100%">
          <el-table-column label="操作" width="100">
            <template #default="{ row }">
              <el-tag :type="tagType(row['操作'])" size="small">{{ row['操作'] }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="代码" label="代码" width="120" />
          <el-table-column prop="名称" label="名称" width="160" />
          <el-table-column prop="打分" label="打分" width="100" />
          <el-table-column prop="排名" label="排名" width="80" />
          <el-table-column prop="现价" label="现价" />
        </el-table>
        <el-alert
          class="mt16"
          type="warning"
          :closable="false"
          title="模型决策支持，不是投资建议；确认按清单成交后，运行 python -m src.decide --no-refresh --apply 更新持仓。"
        />
      </el-card>
    </template>
    <el-empty v-else-if="!loading" description="暂无决策清单：交易日 14:30 到「数据管道」运行 每日决策(decide)" />
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import api from '../api'

const loading = ref(true)
const decision = ref(null)
const positions = ref(null)

const tagType = (op) => ({ 买入: 'danger', 卖出: 'success', 持有: 'info' }[op] || 'info')

onMounted(async () => {
  try {
    decision.value = await api.get('/decision/latest')
    positions.value = decision.value.positions || null
  } catch {
    /* 无决策数据时显示空态 */
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.mb16 {
  margin-bottom: 16px;
}
.mt16 {
  margin-top: 16px;
}
.pos-line {
  margin-bottom: 12px;
  font-size: 13px;
  color: #606266;
}
</style>
