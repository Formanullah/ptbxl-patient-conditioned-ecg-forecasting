param([string]$Dest, [string]$List, [string]$Base)
$ProgressPreference = 'SilentlyContinue'
$files = Get-Content $List
$files | ForEach-Object { Split-Path (Join-Path $Dest $_) } | Sort-Object -Unique | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }
$files | ForEach-Object -Parallel {
  $out = Join-Path $using:Dest $_
  if (-not (Test-Path $out)) { Invoke-WebRequest -Uri ($using:Base + $_) -OutFile $out -UseBasicParsing }
} -ThrottleLimit 32
