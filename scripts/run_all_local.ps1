<#
.SYNOPSIS
    本地依次训练 5 组实验（E1/E2/E3 + A1/A2）。

.DESCRIPTION
    等价于 scripts/run_all.sh，但使用本地数据路径。已有 runs/<name>/summary.json
    的实验默认跳过；-Force 可强制重跑。

.EXAMPLE
    .\scripts\run_all_local.ps1

.EXAMPLE
    .\scripts\run_all_local.ps1 -Epochs 5
#>
[CmdletBinding()]
param(
    [string]$DataDir = 'data/iwslt17_zh_en',
    [string]$SpModel = 'prepared/spm_zh_en.model',
    [string]$RunsDir = 'runs',
    [int]$Epochs = 0,
    [switch]$Force,
    [string]$Device = 'auto'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

$configs = @(
    'configs/lstm.yaml',
    'configs/transformer.yaml',
    'configs/transformer_improved.yaml',
    'configs/ablation_no_pe.yaml',
    'configs/ablation_1head.yaml'
)

foreach ($cfg in $configs) {
    $name = [IO.Path]::GetFileNameWithoutExtension($cfg)
    $summary = Join-Path (Join-Path $RunsDir $name) 'summary.json'
    if ((-not $Force) -and (Test-Path $summary)) {
        Write-Host "skip $name : $summary exists (use -Force to rerun)" -ForegroundColor Yellow
        continue
    }
    Write-Host ('=' * 70) -ForegroundColor DarkGray
    Write-Host "training $name"
    Write-Host ('=' * 70) -ForegroundColor DarkGray

    $argv = @(
        '-Config', $cfg,
        '-RunName', $name,
        '-DataDir', $DataDir,
        '-SpModel', $SpModel,
        '-RunsDir', $RunsDir,
        '-Device', $Device
    )
    if ($Epochs -gt 0) { $argv += @('-Epochs', $Epochs) }
    & (Join-Path $PSScriptRoot 'train_local.ps1') @argv
}

Write-Host "all runs finished. results in $RunsDir" -ForegroundColor Green
