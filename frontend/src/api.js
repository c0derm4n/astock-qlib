import axios from 'axios'
import { ElMessage } from 'element-plus'

const api = axios.create({ baseURL: '/api', timeout: 30000, headers: { 'X-Local-Request': '1' } })

// 统一错误提示；业务码直接取响应 data
api.interceptors.response.use(
  (resp) => resp.data,
  (err) => {
    if (!axios.isCancel(err) && !err.config?.silentErrors) {
      const detail = err.response?.data?.detail
      const msg = Array.isArray(detail)
        ? detail.map((item) => item.msg).join('；')
        : detail || err.message || '请求失败'
      ElMessage.error(msg)
    }
    return Promise.reject(err)
  },
)

export default api
