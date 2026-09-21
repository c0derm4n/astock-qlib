<template>
  <div v-loading="loading">
    <template v-if="decision">
      <el-alert
        class="mb16"
        type="info"
        :closable="false"
        :title="decision.legacy ? '历史盘中清单已停用，请重新生成收盘后模拟决策' : `${decision.date} 收盘信号 → ${decision.execution_date} 开盘模拟订单；QDII 不参与`"
      />
      <el-card>
        <el-alert v-if="positions && !positions.valid" type="error" :closable="false" :title="positions.error" />
        <div v-else-if="positions" class="pos-line">已核对持仓：{{ positions.positions.length ? positions.positions.join('、') : '空仓' }}；现金 {{ positions.cash?.toFixed(2) }} 元</div>
        <el-table v-if="!decision.legacy" :data="decision.rows" stripe style="width: 100%">
          <el-table-column label="操作" width="100">
            <template #default="{ row }">
              <el-tag :type="tagType(row['操作'])" size="small">{{ row['操作'] }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="代码" label="代码" width="120" />
          <el-table-column prop="名称" label="名称" width="160" />
          <el-table-column prop="份额" label="订单份额" width="100" />
          <el-table-column prop="参考收盘价" label="参考收盘价" width="120" />
          <el-table-column prop="目标权重" label="目标权重" />
        </el-table>
        <el-alert
          class="mt16"
          type="warning"
          :closable="false"
          title="这是模拟订单建议，不代表已成交。只有导入带成交编号、份额、价格和费用的回报，才会更新账户。"
        />
      </el-card>
    </template>
    <el-empty v-else-if="!loading" description="暂无模拟决策：先核对账户，收盘数据完成后运行模拟决策" />
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
