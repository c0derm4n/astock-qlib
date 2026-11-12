export const VALUATION_BAND_KIND = 'valuation_bands'
export const DIVIDEND_BAND_KIND = 'dividend_band'
export const ACTIVE_STRATEGY_IDS = ['csi300_pe_equal', 'trend200_vol15', 'dividend_band_a']
export const isIndexStrategy = strategy => [strategy?.kind, strategy?.definition?.kind].includes('index_benchmark')
export const presentStrategies = strategies => ACTIVE_STRATEGY_IDS.flatMap(id => (strategies || []).filter(s => s.id === id)).map(presentStrategy)
export const DEFAULT_ACCOUNT_PREVIEW_REASON = '估值分批策略需要批次及冷却历史，暂不支持只凭现金/份额试算'

export const isValuationStrategy = strategy => [strategy?.kind, strategy?.definition?.kind].includes(VALUATION_BAND_KIND)
  || strategy?.id === 'csi300_pe_equal'

export const isDividendBandStrategy = strategy => [strategy?.kind, strategy?.definition?.kind].includes(DIVIDEND_BAND_KIND)
  || strategy?.id === 'dividend_band_a'

export function strategyDisplayName(strategy) {
  return isDividendBandStrategy(strategy) ? '积极策略 · 红利低波＋国债' : strategy?.name || '正式策略'
}

export function presentStrategy(strategy) {
  if (!strategy || strategy.name === strategyDisplayName(strategy)) return strategy
  return { ...strategy, name: strategyDisplayName(strategy) }
}

export const isValuationResult = kind => ['valuation', VALUATION_BAND_KIND].includes(kind)
export const isDividendBandResult = kind => kind === DIVIDEND_BAND_KIND
export const executionBasisLabel = (basis, fallback = '下一交易日收盘（15:00 窗口）模拟执行') => ({
  next_session_close: '下一交易日收盘（15:00 窗口）模拟执行',
  next_session_open: '下一交易日开盘（09:30 窗口）模拟执行',
}[basis] || basis || fallback)

export const accountPreviewAvailable = strategy => !isDividendBandStrategy(strategy)
  && strategy?.capabilities?.account_preview !== false

export const accountPreviewReason = strategy => strategy?.capabilities?.reason
  || (isDividendBandStrategy(strategy)
    ? '指数等值仓位不能直接换算成ETF整数委托，因此不提供账户试算'
    : DEFAULT_ACCOUNT_PREVIEW_REASON)

export function resolveStrategyId(requested, strategies, fallback = 'csi300_pe_equal') {
  const ids = Array.isArray(strategies) ? strategies.map(strategy => strategy?.id).filter(Boolean) : []
  if (requested && ids.includes(requested)) return requested
  if (ids.includes(fallback)) return fallback
  return ids[0] || fallback
}

export function strategyNavLabel(strategy) {
  const labels = {
    trend200_vol15: '平衡策略',
    csi300_pe_equal: '稳健策略',
    dividend_band_a: '积极策略',
  }
  return labels[strategy?.id] || strategy?.name || '正式策略'
}

export function strategyEyebrow(strategy) {
  if (isIndexStrategy(strategy)) return 'DIVIDEND TOTAL RETURN'
  if (isDividendBandStrategy(strategy)) return 'DIVIDEND BAND'
  if (isValuationStrategy(strategy)) return 'VALUATION BANDS'
  return strategy?.id === 'trend200_vol15' ? 'TREND & ALLOCATION' : 'CORE & SATELLITE'
}

export function valuationMetricLabel(metric) {
  const labels = {
    PE_TTM_EQUAL: 'PE TTM（ETF.run 等权）',
    pe_ttm: 'PE TTM',
  }
  return labels[metric] || metric || '估值指标'
}

export function valuationCaveat(strategy) {
  return 'PE 等权估值为历史重建结果，并非 PIT 快照，也不能视作整体法估值。'
}
