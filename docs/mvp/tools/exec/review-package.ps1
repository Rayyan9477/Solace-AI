<#
.SYNOPSIS
  Write a review package (changed paths + stat + full diff with context) between two checkpoint trees.
.DESCRIPTION
  Replaces the commit-based `review-package BASE HEAD` of subagent-driven development: BASE and HEAD are tree ids
  printed by checkpoint.ps1. Parallel lanes share one working tree, so pass -Paths with the task's declared files
  to keep other lanes' edits out of the package. The package also lists every path that changed anywhere between
  the two trees (section "All changed paths"), so the controller can detect edits outside every running lane's
  declared files. Prints the path of the package file.
.EXAMPLE
  & D:\tmp\mvp-exec\tools\review-package.ps1 -Base <tree> -Head <tree> -Name WS-03.2 -Paths services/safety_service/src/main.py,services/safety_service/tests/test_boot.py
#>
param(
    [Parameter(Mandatory = $true)][string]$Base,
    [Parameter(Mandatory = $true)][string]$Head,
    [Parameter(Mandatory = $true)][string]$Name,
    [string[]]$Paths = @(),
    [string]$Repo = "D:\Repo\Contextual-Chatbot",
    [string]$Root = "D:\tmp\mvp-exec"
)
$ErrorActionPreference = "Stop"
$dir = Join-Path $Root "reviews"
New-Item -ItemType Directory -Force $dir | Out-Null
$out = Join-Path $dir ("{0}-{1}.diff.md" -f $Name, (Get-Date).ToString("yyyyMMdd-HHmmss"))
$all = git -C $Repo diff --name-status $Base $Head
if ($Paths.Count -gt 0) {
    $stat = git -C $Repo diff --stat $Base $Head -- $Paths
    $diff = git -C $Repo diff -U10 $Base $Head -- $Paths
    $scope = "Declared paths: " + ($Paths -join ", ")
}
else {
    $stat = git -C $Repo diff --stat $Base $Head
    $diff = git -C $Repo diff -U10 $Base $Head
    $scope = "Declared paths: (whole tree)"
}
@("# Review package: $Name", "", "Base tree: $Base", "Head tree: $Head", $scope, "", "## All changed paths (whole tree, includes other lanes)", '```') + $all +
  @('```', "", "## Stat (declared paths)", '```') + $stat + @('```', "", "## Diff (declared paths)", '```diff') + $diff + @('```') |
  Set-Content -Encoding utf8 $out
Write-Output $out
