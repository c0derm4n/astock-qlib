param(
    [ValidateRange(1024,65535)][int]$Port = 8000,
    [switch]$SkipBuild,
    [switch]$Refresh,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw '缺少项目虚拟环境。请先创建 .venv 并安装 requirements.txt、backend/requirements.txt。'
}
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
    throw "端口 $Port 已被使用，请用 -Port 8001 等空闲端口启动。"
}
if (-not $SkipBuild -or -not (Test-Path -LiteralPath 'frontend\dist\index.html')) {
    $npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npmCommand) { throw '缺少 Node.js/npm，请先安装 Node.js LTS。' }
    Push-Location -LiteralPath 'frontend'
    try {
        if (-not (Test-Path -LiteralPath 'node_modules')) {
            & $npmCommand.Source ci
            if ($LASTEXITCODE -ne 0) { throw '前端依赖安装失败。' }
        }
        & $npmCommand.Source run build
        if ($LASTEXITCODE -ne 0) { throw '前端构建失败。' }
    } finally { Pop-Location }
}
$needsSnapshot = -not (Test-Path -LiteralPath 'output\strategies\current.json')
if (-not $needsSnapshot) {
    $freshness = & $pythonExe -c "from src.formal_service import snapshot_needs_refresh; print(int(snapshot_needs_refresh()))"
    $needsSnapshot = ($LASTEXITCODE -ne 0) -or ($freshness -ne '0')
}
if ($Refresh -or $needsSnapshot) {
    & $pythonExe -m src.formal_service --fetch
    if ($LASTEXITCODE -ne 0) { throw '正式策略构建失败；详情见终端。' }
}
$localUrl = "http://127.0.0.1:$Port"
Write-Host "ETF三策略服务：$localUrl"
Write-Host '启动后可点击页面的“更新行情与建议”。按 Ctrl+C 停止服务。'
if (-not $NoBrowser) {
    Start-Process $localUrl
}
& $pythonExe -m uvicorn backend.main:app --host 127.0.0.1 --port $Port --workers 1
exit $LASTEXITCODE
