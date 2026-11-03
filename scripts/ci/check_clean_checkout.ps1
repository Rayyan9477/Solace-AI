<#
.SYNOPSIS
  Build and probe every registered image from `git archive <tree>` (MVP-008, D-17: no commits, trees only).
.EXAMPLE
  pwsh -File scripts\ci\check_clean_checkout.ps1 -Tree 41ed16e -Compose
#>
param(
    [Parameter(Mandatory = $true)][string]$Tree,
    [string]$Tag = "",
    [string]$Out = "",
    [switch]$Compose,
    [switch]$Tests,
    [switch]$RemoveImages,
    [switch]$AllowRegistryDrift   # explicit proofs of older trees; the default requires the registries to match
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = "D:\tmp\mvp-exec\venv\Scripts\python.exe"   # D-20
$arguments = @((Join-Path $repo "scripts\ci\clean_checkout.py"), "--tree", $Tree)
if ($Tag) { $arguments += @("--tag", $Tag) }
if ($Out) { $arguments += @("--out", $Out) }
if (-not $AllowRegistryDrift) { $arguments += "--require-registry-match" }
if ($Compose) { $arguments += "--compose" }
if ($Tests) { $arguments += "--tests" }
if ($RemoveImages) { $arguments += "--remove-images" }
& $python @arguments
exit $LASTEXITCODE
