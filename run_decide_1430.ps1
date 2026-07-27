# 交易日 14:30 盘中决策：由 Windows 任务计划调用（也可手工运行）。
# 注册任务计划（管理员或当前用户均可，周一到周五 14:30 触发；周末由脚本自行跳过）：
#   schtasks /Create /TN "ETF轮动1430决策" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 14:30 `
#     /TR "pwsh -NoProfile -ExecutionPolicy Bypass -File e:\tcl_expert_llm\astock-qlib\run_decide_1430.ps1"
# 取消：schtasks /Delete /TN "ETF轮动1430决策" /F
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# 如需 Tushare NAV 数据，请在此处或系统环境变量中配置：
# $env:TUSHARE_TOKEN = "你的token"

$log = Join-Path $PSScriptRoot ("output\decide_{0:yyyyMMdd}.log" -f (Get-Date))
& "$PSScriptRoot\.venv\Scripts\python.exe" -m src.decide *>&1 | Tee-Object -FilePath $log
exit $LASTEXITCODE
