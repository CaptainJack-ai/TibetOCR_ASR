$ErrorActionPreference = 'Stop'
$projectPath = $PSScriptRoot
$pythonPath = Join-Path (Split-Path $projectPath -Parent) 'LargeModel_Test_Related\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $pythonPath = (Get-Command python -ErrorAction Stop).Source
}
Set-Location -LiteralPath $projectPath
& $pythonPath -X utf8 gui/server.py --offline --port 8765
