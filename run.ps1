# Windows: launch from PowerShell as described in README.md. No Python required.
param(
    [ValidateSet('app', 'live', 'research', 'check')]
    [string]$Mode = 'app'
)
$ErrorActionPreference = 'Stop'
$exitCode = 0
try {
    Set-Location -LiteralPath $PSScriptRoot
    Write-Host 'Portfolio Engine'
    Write-Host 'First launch downloads Python 3.11 and app dependencies; later launches reuse them.'
    Write-Host 'Setup stays in .runtime inside this folder. No administrator access is needed.'

    $uvBin = Join-Path $PSScriptRoot '.runtime\uv\uv.exe'
    if (-not (Test-Path -LiteralPath $uvBin)) {
        $existingUv = Get-Command uv.exe -CommandType Application -ErrorAction SilentlyContinue
        if ($existingUv) {
            $uvBin = $existingUv.Source
        } else {
            Write-Host 'Downloading the uv setup tool from astral.sh...'
            $env:UV_UNMANAGED_INSTALL = Join-Path $PSScriptRoot '.runtime\uv'
            New-Item -ItemType Directory -Force -Path $env:UV_UNMANAGED_INSTALL | Out-Null
            # Official Astral installer. TLS 1.2 supports Windows PowerShell 5.1.
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            Invoke-RestMethod 'https://astral.sh/uv/install.ps1' | Invoke-Expression
        }
    }
    if (-not (Test-Path -LiteralPath $uvBin)) { throw 'The uv setup tool was not installed successfully.' }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot '.runtime\python'
    $env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.runtime\cache'
    $env:UV_PYTHON_DOWNLOADS = 'automatic'
    $uvArgs = @('run', '--isolated', '--no-project', '--managed-python', '--python', '3.11',
                '--with-requirements', (Join-Path $PSScriptRoot 'requirements.txt'))
    switch ($Mode) {
        'app' {
            Write-Host 'The dashboard will open in your browser at http://localhost:8501.'
            Write-Host 'Keep this window open. Press Control+C here to stop the app.'
            & $uvBin @uvArgs python -m streamlit run ui/app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
        }
        'live' { & $uvBin @uvArgs python main_live.py }
        'research' { & $uvBin @uvArgs python main_research.py }
        'check' { & $uvBin @uvArgs python -c "import sys, streamlit, plotly, portfolio_decision, main_live, main_research; from sklearn.covariance import LedoitWolf; print('Setup verified. Python ' + sys.version.split()[0] + '; app, engine and covariance dependencies OK.')" }
    }
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) { throw "Launcher exited with code $exitCode." }
} catch {
    Write-Host "`nCould not start Portfolio Engine: $_" -ForegroundColor Red
    Write-Host 'Check your internet connection and that this folder is writable.'
    if ($exitCode -eq 0) { $exitCode = 1 }
    Read-Host 'Press Enter to close' | Out-Null
}
exit $exitCode
