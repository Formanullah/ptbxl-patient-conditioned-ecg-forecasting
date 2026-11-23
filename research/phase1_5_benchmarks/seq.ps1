param([string]$Dest, [string]$List, [string]$Base)
$ProgressPreference = 'SilentlyContinue'
foreach ($f in Get-Content $List) {
  $out = Join-Path $Dest $f; New-Item -ItemType Directory -Force -Path (Split-Path $out) | Out-Null
  Invoke-WebRequest -Uri ($Base + $f) -OutFile $out -UseBasicParsing
}
