<#
.SYNOPSIS
  Commit-free checkpoint of the whole working tree (owner rule: no git commits).
.DESCRIPTION
  Stages the working tree into a TEMPORARY index (GIT_INDEX_FILE), writes a git tree object and prints its id.
  No commit, no ref, no branch, and the real index is never touched. Ignored files (.env, caches) are excluded
  by .gitignore. The tree id is appended to D:\tmp\mvp-exec\snapshots\snapshots.log with the task label.
.EXAMPLE
  pwsh -File D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-03.2:after"
#>
param(
    [Parameter(Mandatory = $true)][string]$Label,
    [string]$Repo = "D:\Repo\Contextual-Chatbot",
    [string]$Root = "D:\tmp\mvp-exec"
)
$ErrorActionPreference = "Stop"
$snapDir = Join-Path $Root "snapshots"
New-Item -ItemType Directory -Force $snapDir | Out-Null
$idx = Join-Path $snapDir (".index-" + [guid]::NewGuid().ToString("N"))
$prev = $env:GIT_INDEX_FILE
try {
    $env:GIT_INDEX_FILE = $idx
    git -C $Repo read-tree HEAD
    if ($LASTEXITCODE -ne 0) { throw "git read-tree failed" }
    git -c core.safecrlf=false -C $Repo add -A
    if ($LASTEXITCODE -ne 0) { throw "git add -A (temporary index) failed" }
    $tree = (git -C $Repo write-tree).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $tree) { throw "git write-tree failed" }
}
finally {
    if ($null -eq $prev) { Remove-Item Env:\GIT_INDEX_FILE -ErrorAction SilentlyContinue } else { $env:GIT_INDEX_FILE = $prev }
    Remove-Item $idx -Force -ErrorAction SilentlyContinue
}
"{0}`t{1}`t{2}" -f (Get-Date).ToString("s"), $Label, $tree | Add-Content -Encoding utf8 (Join-Path $snapDir "snapshots.log")
Write-Output $tree
