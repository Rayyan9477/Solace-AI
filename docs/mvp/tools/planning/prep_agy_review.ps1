<#
  Export a commit-free snapshot of the working tree (plans included) into N scratch workspaces for AGY.
  AGY has no read-only mode, so it never sees the real repo. Prints the tree id and the workspace paths.
#>
param(
    [int]$Groups = 6,
    [string]$Label = "plan:detail-for-agy",
    [string]$Out = "D:\tmp\mvp-plan\agy-review"
)
$ErrorActionPreference = "Stop"
$repo = "D:\Repo\Contextual-Chatbot"
$tree = (& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label $Label | Select-Object -Last 1).Trim()
if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force $Out | Out-Null
$zip = Join-Path $Out "snap.zip"
git -C $repo archive --format=zip -o $zip $tree
if ($LASTEXITCODE -ne 0) { throw "git archive failed" }
git -C $repo status --porcelain=v1 -uall | Set-Content -Encoding utf8 (Join-Path $Out "real-repo-status-before.txt")
for ($i = 1; $i -le $Groups; $i++) {
    $ws = Join-Path $Out ("g{0}" -f $i)
    Expand-Archive -Path $zip -DestinationPath $ws -Force
    $pl = Join-Path $ws "_planning"
    New-Item -ItemType Directory -Force $pl | Out-Null
    Copy-Item D:\tmp\mvp-plan\skeleton.json, D:\tmp\mvp-plan\register_compiled.json, D:\tmp\mvp-plan\check_report.md, D:\tmp\mvp-plan\agy_instructions.md -Destination $pl
    if (Test-Path D:\tmp\mvp-plan\task_index.json) { Copy-Item D:\tmp\mvp-plan\task_index.json -Destination $pl }
}
Write-Output "tree=$tree"
Get-ChildItem $Out -Directory | ForEach-Object { Write-Output $_.FullName }
