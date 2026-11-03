<#
.SYNOPSIS
  Restore the given paths in the working tree to their content in a checkpoint tree (no commit, real index untouched).
.DESCRIPTION
  Paths that did not exist in the checkpoint tree are deleted. Use after a failed or rejected task, only for the
  task's own declared files.
.EXAMPLE
  pwsh -File D:\tmp\mvp-exec\tools\rollback.ps1 -Tree <tree> -Paths services/safety_service/src/main.py,services/safety_service/tests/test_x.py
#>
param(
    [Parameter(Mandatory = $true)][string]$Tree,
    [Parameter(Mandatory = $true)][string[]]$Paths,
    [string]$Repo = "D:\Repo\Contextual-Chatbot"
)
$ErrorActionPreference = "Stop"
foreach ($p in $Paths) {
    # ls-tree prints the path when it exists in the tree and nothing otherwise (no stderr, which
    # Windows PowerShell 5.1 would turn into a terminating error under $ErrorActionPreference=Stop).
    $inTree = git -C $Repo ls-tree --name-only $Tree -- $p
    if ($inTree) {
        git -C $Repo restore --source=$Tree --worktree -- $p
        if ($LASTEXITCODE -ne 0) { throw "restore failed: $p" }
        Write-Output "restored $p"
    }
    else {
        $full = Join-Path $Repo $p
        if (Test-Path $full) { Remove-Item -Force $full; Write-Output "removed new file $p" }
    }
}
