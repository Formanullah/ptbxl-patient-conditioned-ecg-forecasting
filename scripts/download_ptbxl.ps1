# Download and verify PTB-XL v1.0.3 (Windows; also runs under PowerShell 7 on Linux/macOS).
# No admin rights needed. Uses uv if installed, otherwise Python 3.9+ from PATH.
# All arguments are passed through, e.g.:
#   .\scripts\download_ptbxl.ps1 --root PTB-XL
#   .\scripts\download_ptbxl.ps1 --root PTB-XL --verify-only
# If PowerShell blocks the script, run it as:
#   powershell -ExecutionPolicy Bypass -File scripts\download_ptbxl.ps1 --root PTB-XL
$ErrorActionPreference = 'Stop'
$script = Join-Path $PSScriptRoot 'download_ptbxl.py'
if (Get-Command uv -ErrorAction SilentlyContinue) {
    & uv run --no-project python $script @args
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 $script @args
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    & python3 $script @args
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python $script @args
} else {
    Write-Error 'Install uv (https://docs.astral.sh/uv/) or Python 3.9+ first.'
}
exit $LASTEXITCODE
