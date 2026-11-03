<#
.SYNOPSIS
  Extract one task's full text from its workstream plan into a brief file (replaces the bash `task-brief` script).
.DESCRIPTION
  Copies from the "### Task <Id>:" heading up to the next "### Task " or "## " heading. Prints the brief path.
.EXAMPLE
  & D:\tmp\mvp-exec\tools\extract-brief.ps1 -Task WS-03.2
#>
param(
    [Parameter(Mandatory = $true)][string]$Task,
    [string]$PlanDir = "D:\Repo\Contextual-Chatbot\docs\superpowers\plans\2026-10-18-mvp-remediation",
    [string]$Root = "D:\tmp\mvp-exec"
)
$ErrorActionPreference = "Stop"
$ws = $Task.Split(".")[0]
$file = Get-ChildItem $PlanDir -Filter "$ws-*.md" | Select-Object -First 1
if (-not $file) { throw "No plan file for $ws in $PlanDir" }
$lines = Get-Content -Encoding utf8 $file.FullName
$start = -1
for ($i = 0; $i -lt $lines.Count; $i++) { if ($lines[$i] -like "### Task ${Task}:*") { $start = $i; break } }
if ($start -lt 0) { throw "Task $Task not found in $($file.Name)" }
$end = $lines.Count
for ($j = $start + 1; $j -lt $lines.Count; $j++) { if ($lines[$j] -like "### Task *" -or $lines[$j] -like "## *") { $end = $j; break } }
$dir = Join-Path $Root "briefs"
New-Item -ItemType Directory -Force $dir | Out-Null
$out = Join-Path $dir "$Task-brief.md"
@("<!-- brief extracted from $($file.Name) -->") + $lines[$start..($end - 1)] | Set-Content -Encoding utf8 $out
Write-Output $out
