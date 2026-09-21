# 旧入口仅保留迁移提示。已有任务计划请改到19:00并使用run_decide_1900.ps1。
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# 如需 Tushare NAV 数据，请在此处或系统环境变量中配置：
# $env:TUSHARE_TOKEN = "你的token"

$log = Join-Path $PSScriptRoot ("output\decide_{0:yyyyMMdd}.log" -f (Get-Date))
Write-Warning "旧14:30入口已停用；请将任务计划改为19:00调用run_decide_1900.ps1。"
& "$PSScriptRoot\.venv\Scripts\python.exe" -m src.decide --paper *>&1 | Tee-Object -FilePath $log
exit $LASTEXITCODE
