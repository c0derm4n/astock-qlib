<template>
  <div>
    <el-card class="mb16">
      <el-form inline @submit.prevent>
        <el-form-item label="交易日">
          <el-date-picker
            v-model="date"
            type="date"
            value-format="YYYY-MM-DD"
            placeholder="默认最新"
            :disabled-date="disabledDate"
            :clearable="true"
          />
        </el-form-item>
        <el-form-item label="TopK">
          <el-input-number v-model="topk" :min="1" :max="50" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="loading" @click="load">查询</el-button>
          <el-button :loading="loading" @click="loadLatest">最新</el-button>
        </el-form-item>
      </el-form>
      <div v-if="actualDate && date && actualDate !== date" class="fallback-hint">
        {{ date }} 非交易日或无数据，已回退到最近交易日 {{ actualDate }}
      </div>
    </el-card>

    <el-card v-if="rows.length">
      <template #header>{{ actualDate }} 模型打分 Top{{ rows.length }} 选ETF清单</template>
      <el-table :data="rows" stripe style="width: 100%">
        <el-table-column prop="rank" label="排名" width="80" sortable />
        <el-table-column prop="code" label="代码" width="140" />
        <el-table-column prop="name" label="名称" width="180" />
        <el-table-column prop="score" label="打分">
          <template #default="{ row }">{{ row.score.toFixed(4) }}</template>
        </el-table-column>
      </el-table>
      <el-alert class="mt16" type="warning" :closable="false" title="模型打分排名，不是投资建议；请结合基本面与风控，务必先模拟盘验证。" />
    </el-card>
    <el-empty v-else-if="!loading" description="暂无数据：请先到「数据管道」运行 训练(train)" />
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import api from '../api'

const date = ref(null)
const topk = ref(6) // 默认与 config.TOPK 对齐，onMounted 时从 /config 同步
const rows = ref([])
const actualDate = ref('')
const loading = ref(false)
const validDates = ref(new Set())

const disabledDate = (d) => {
  // 拿到预测日期表后只允许选有数据的日子；接口本身也支持回退
  if (!validDates.value.size) return false
  const s = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
  return !validDates.value.has(s)
}

async function load() {
  loading.value = true
  try {
    const data = await api.get('/picks', { params: { date: date.value || undefined, topk: topk.value } })
    rows.value = data.rows
    actualDate.value = data.date
  } catch {
    rows.value = []
  } finally {
    loading.value = false
  }
}

function loadLatest() {
  date.value = null
  load()
}

onMounted(async () => {
  try {
    const [cfg, dates] = await Promise.allSettled([api.get('/config'), api.get('/picks/dates')])
    if (cfg.status === 'fulfilled' && cfg.value.topk) topk.value = cfg.value.topk
    if (dates.status === 'fulfilled') validDates.value = new Set(dates.value)
    await load()
  } catch {
    /* 无预测数据时显示空态 */
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
.fallback-hint {
  color: #e6a23c;
  font-size: 13px;
}
</style>
