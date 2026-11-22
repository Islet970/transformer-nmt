<#
.SYNOPSIS
    在本地（Windows + CUDA）训练一个模型。

.DESCRIPTION
    对 `python train.py` 的薄封装：默认使用本仓库下载好的数据与 tokenizer，
    自动检查六个数据文件和 spm 模型是否存在。

.EXAMPLE
    .\scripts\train_local.ps1 -Config configs/transformer_improved.yaml

.EXAMPLE
    # 20 步 smoke test
    .\scripts\train_local.ps1 -RunName smoke -Epochs 1 -MaxSteps 20

.EXAMPLE
    # 断点续训
    .\scripts\train_local.ps1 -Config configs/transformer_improved.yaml -Resume
#>
[CmdletBinding()]
param(
    [string]$Config = 'configs/transformer_improved.yaml',
    [string]$RunName = '',
    [string]$DataDir = 'data/iwslt17_zh_en',
    [string]$SpModel = 'prepared/spm_zh_en.model',
    [string]$RunsDir = 'runs',
    [int]$Epochs = 0,
    [int]$BatchSize = 0,
    [int]$MaxSteps = 0,
    [switch]$Resume,
    [string]$Device = 'auto',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Extra
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
    if (-not (Test-Path $Config)) { throw "config not found: $Config" }
    if (-not $RunName) { $RunName = [IO.Path]::GetFileNameWithoutExtension($Config) }

    foreach ($name in 'train.zh', 'train.en', 'valid.zh', 'valid.en', 'test.zh', 'test.en') {
        $path = Join-Path $DataDir $name
        if (-not (Test-Path $path)) { throw "missing data file: $path" }
    }
    if (-not (Test-Path $SpModel)) {
        throw "tokenizer not found: $SpModel`nrun: python -m data.prepare --data-dir $DataDir --out-dir prepared"
    }

    $argv = @(
        'train.py',
        '--config', $Config,
        '--data-dir', $DataDir,
        '--sp-model', $SpModel,
        '--output-dir', $RunsDir,
        '--run-name', $RunName,
        '--device', $Device
    )
    if ($Epochs -gt 0) { $argv += @('--epochs', $Epochs) }
    if ($BatchSize -gt 0) { $argv += @('--batch-size', $BatchSize) }
    if ($MaxSteps -gt 0) { $argv += @('--max-steps', $MaxSteps) }
    if ($Resume) { $argv += '--resume' }
    if ($Extra) { $argv += $Extra }

    Write-Host ("python " + ($argv -join ' ')) -ForegroundColor Cyan
    & python @argv
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
