<template>
  <div>
    <el-card class="mb16">
      <template #header>
        <span>管道步骤（与 README 快速开始一致）</span>
        <el-button class="run-all" type="primary" :disabled="status.running || submitting" @click="run(null)">
          一键运行 fetch → train
        </el-button>
      </template>
      <el-row :gutter="16">
        <el-col :span="8" v-for="s in steps" :key="s.key">
          <el-card shadow="hover">
            <div class="step-head">
              <span class="step-name">{{ s.name }}</span>
              <el-tag :type="tagType(status.steps[s.key])" size="small">{{ tagText(status.steps[s.key]) }}</el-tag>
            </div>
            <div class="step-cmd">python -m {{ s.module }}</div>
            <div class="step-desc">{{ s.desc }}</div>
            <el-button size="small" :disabled="status.running || submitting" @click="run([s.key])">运行此步</el-button>
          </el-card>
        </el-col>
      </el-row>
      <div class="status-line">
        <span v-if="status.started_at">开始：{{ status.started_at }}</span>
        <span v-if="status.finished_at">　结束：{{ status.finished_at }}</span>
        <span v-if="status.error" class="err">　{{ status.error }}</span>
      </div>
    </el-card>

    <el-card>
      <template #header>
        <span>运行日志</span>
        <span class="card-sub">完整日志见 output/pipeline_logs/</span>
      </template>
      <pre ref="logEl" class="log">{{ status.log_tail.join('\n') || '（暂无日志）' }}</pre>
    </el-card>
  </div>
</template>

<script setup>
import { nextTick, onMounted, onUnmounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'

const steps = [
  { key: 'fetch', name: '1. 抓取行情', module: 'src.fetch_data', desc: '自动选择授权或免费源；核对历史重叠与复权后增量更新' },
  { key: 'train', name: '2. 训练验收', module: 'src.train', desc: '滚动历史研究、对照与压力测试，生成独立生产模型' },
  { key: 'decide', name: '3. 模拟决策', module: 'src.decide --paper', desc: '19:00 后用正式收盘数据，生成下一交易日开盘模拟订单' },
]

const status = reactive({ running: false, current_step: null, steps: {}, started_at: null, finished_at: null, error: null, log_tail: [] })
const logEl = ref(null)
const submitting = ref(false)
let timer = null
let request = null
let controller = null
let disposed = false

const tagType = (s) => ({ running: 'warning', done: 'success', failed: 'danger' }[s] || 'info')
const tagText = (s) => ({ pending: '等待中', running: '运行中', done: '成功', failed: '失败', skipped: '已跳过' }[s] || '未运行')

async function poll() {
  if (disposed) return
  if (request) return request
  clearTimeout(timer)
  controller = new AbortController()
  request = (async () => {
    try {
      const data = await api.get('/pipeline/status', { signal: controller.signal, silentErrors: true })
      if (disposed) return
      Object.assign(status, data)
      scrollLog()
    } catch {
      /* 轮询失败时静默重试，避免持续弹出错误提示。 */
    } finally {
      request = null
      controller = null
      if (!disposed) timer = setTimeout(poll, status.running ? 2000 : 10000)
    }
  })()
  return request
}

async function scrollLog() {
  await nextTick()
  if (logEl.value) logEl.value.scrollTop = logEl.value.scrollHeight
}

async function run(stepKeys) {
  if (submitting.value || status.running) return
  submitting.value = true
  try {
    await api.post('/pipeline/run', { steps: stepKeys })
    if (disposed) return
    ElMessage.success('任务已启动')
    // 等待已有查询结束后重新获取启动后的状态。
    if (request) await request
    await poll()
  } catch {
    /* 409/400 已由拦截器提示 */
  } finally {
    submitting.value = false
  }
}

onMounted(poll)
onUnmounted(() => {
  disposed = true
  clearTimeout(timer)
  controller?.abort()
})
</script>

<style scoped>
.mb16 {
  margin-bottom: 16px;
}
.run-all {
  float: right;
}
.step-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.step-name {
  font-weight: 600;
}
.step-cmd {
  font-family: Consolas, monospace;
  font-size: 12px;
  color: #606266;
  background: #f5f7fa;
  padding: 4px 8px;
  border-radius: 4px;
  margin-bottom: 8px;
}
.step-desc {
  font-size: 13px;
  color: #909399;
  min-height: 36px;
  margin-bottom: 8px;
}
.status-line {
  margin-top: 12px;
  font-size: 13px;
  color: #606266;
}
.err {
  color: #f56c6c;
}
.card-sub {
  float: right;
  font-size: 12px;
  color: #909399;
  font-weight: normal;
}
.log {
  background: #1e1e1e;
  color: #d4d4d4;
  font-size: 12px;
  line-height: 1.6;
  padding: 12px;
  border-radius: 4px;
  height: 380px;
  overflow: auto;
  margin: 0;
  white-space: pre-wrap;
  word-break: break-all;
}
</style>
