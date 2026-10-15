import { onMounted, onBeforeUnmount, ref, shallowRef } from 'vue'
import api from '../api'

export function requestError(error, fallback = '无法读取正式策略快照，请确认本地服务已启动。') {
  const detail = error.response?.data?.detail
  return Array.isArray(detail) ? detail.map(item => item.msg).join('；') : detail || fallback
}

// 每个开发页面只读取同一份正式双策略运行，后续明细查询绑定 run_id。
export function useFormalSnapshot() {
  const snapshot = shallowRef(null), loading = ref(true), error = ref('')
  let controller, revision = 0, disposed = false
  async function load() {
    const current = ++revision
    controller?.abort()
    controller = new AbortController()
    loading.value = true; error.value = ''
    try {
      const result = await api.get('/development/overview', { signal: controller.signal, silentErrors: true })
      if (!disposed && current === revision) snapshot.value = result
    } catch (err) {
      if (!disposed && current === revision && err.code !== 'ERR_CANCELED') error.value = requestError(err)
    } finally { if (!disposed && current === revision) loading.value = false }
  }
  onMounted(load)
  onBeforeUnmount(() => { disposed = true; revision++; controller?.abort() })
  return { snapshot, loading, error, load }
}

export const number = (value, digits = 2) => value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toLocaleString('zh-CN', { minimumFractionDigits: digits, maximumFractionDigits: digits })
export const percent = value => value == null || !Number.isFinite(Number(value)) ? '—' : `${(value * 100).toFixed(2)}%`
export const sideName = side => ({ buy: '买入', sell: '卖出', hold: '保持' })[side] || side || '—'
export const formatTime = value => value ? new Date(value).toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', hour12: false }) : '—'
export const foldRows = value => Array.isArray(value) ? value : value?.folds || []
