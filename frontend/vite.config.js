import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      // 开发模式把 /api 代理到 FastAPI 后端
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
