import { isDividendBandStrategy, isValuationStrategy } from './strategyPresentation.js'

const canonicalCode = code => String(code || '').replace(/^(SH|SZ)(?=\d)/, '')

export function allocationWeeks(strategy) {
  const source = strategy.weekly_allocation?.length ? strategy.weekly_allocation : strategy.weekly_holdings || []
  return source.map(week => {
    const date = week.date || week.week_start
    const account = Number(week.account)
    return {
      ...week,
      date,
      week_start: week.week_start || date,
      week_end: week.week_end || date,
      cash_weight: week.cash_weight ?? (account > 0 ? Number(week.cash) / account : null),
      receivable_weight: week.receivable_weight ?? (account > 0 ? Number(week.dividend_receivable || 0) / account : null),
    }
  }).sort((a, b) => a.week_start.localeCompare(b.week_start)).slice(-50)
}

export function allocationWeight(week, code) {
  return (week.holdings || [])
    .filter(holding => canonicalCode(holding.code) === code || canonicalCode(holding.symbol) === code)
    .reduce((sum, holding) => sum + Number(holding.weight), 0)
}

export const totalAllocationWeight = week => (week.holdings || []).reduce((sum, holding) => sum + Number(holding.weight), 0)

export function allocationAssets(strategy, weeks) {
  if (isDividendBandStrategy(strategy)) return [
    { code: 'H20269', name: '红利低波', color: '#a84c54' },
    { code: '511260', name: '十年国债', color: '#c29447' },
  ]
  if (isValuationStrategy(strategy)) return [{ code: '510300', name: '沪深300ETF', color: '#3975ca' }]
  return [
    { code: '515880', name: '通信ETF', color: '#3975ca' },
    { code: '518880', name: '黄金ETF', color: '#c29447' },
    ...(weeks.some(week => allocationWeight(week, '510300') > 0)
      ? [{ code: '510300', name: '沪深300（资格替代）', color: '#8395a9' }] : []),
  ]
}
