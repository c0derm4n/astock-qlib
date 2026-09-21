# 收盘正式数据完成后生成下一交易日开盘模拟订单；不自动下单。
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$taskLog = Join-Path $PSScriptRoot ("output\decide_{0:yyyyMMdd}.log" -f (Get-Date))
& "$PSScriptRoot\.venv\Scripts\python.exe" -m src.decide --paper *>&1 | Tee-Object -FilePath $taskLog
exit $LASTEXITCODE
