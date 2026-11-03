<#
.SYNOPSIS
  Operator helper for the E2E compose stack of one lane (MVP-039, WS-05.4). TEST ONLY.

.DESCRIPTION
  Runs "python -m solace_testing.e2e <action>", which uses exactly the compose command of the e2e_stack
  fixture (tests/e2e/conftest.py): project solace-e2e-<lane>, the per-run env file from
  scripts/test/gen_e2e_env.py, docker-compose.yml [+ docker-compose.prod.yml] + docker-compose.test.yml, and a
  cleaned environment for compose.

    up      generate fresh secrets, build, start and wait for the stack; it is kept until "down"
    down    remove the containers, networks, volumes and the run directory (with its secrets)
    logs    the container logs (of -Service, or of every service)
    ps      the containers of the stack

  Secrets are never printed; "up" prints the edge, mailpit and fake-llm URLs only.

.EXAMPLE
  .\scripts\test\e2e_stack.ps1 up -E2EProfile strict -E2EEnvironment production -Lane b
  .\scripts\test\e2e_stack.ps1 logs -Lane b -Service user-service
  .\scripts\test\e2e_stack.ps1 down -Lane b
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("up", "down", "logs", "ps")]
    [string]$Action,

    [ValidateSet("dev", "strict")]
    [string]$E2EProfile = "dev",

    [ValidateSet("", "development", "staging", "production")]
    [string]$E2EEnvironment = "",

    [ValidatePattern("^[a-z0-9][a-z0-9-]{0,15}$")]
    [string]$Lane = "a",

    [string]$Service = "",

    [string]$RunDir = "",

    # The interpreter with the test dependencies (default: $env:SOLACE_PYTHON, else python on PATH).
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Python) {
    if ($env:SOLACE_PYTHON) { $Python = $env:SOLACE_PYTHON } else { $Python = "python" }
}
if ($Service -and $Action -ne "logs") {
    throw "-Service applies to logs only"
}

$names = @("SOLACE_E2E_PROFILE", "SOLACE_E2E_ENVIRONMENT", "SOLACE_E2E_LANE", "SOLACE_E2E_RUN_DIR", "PYTHONPATH")
$saved = @{}
foreach ($name in $names) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
try {
    $env:SOLACE_E2E_PROFILE = $E2EProfile
    if ($E2EEnvironment) {
        $env:SOLACE_E2E_ENVIRONMENT = $E2EEnvironment
    } else {
        Remove-Item Env:SOLACE_E2E_ENVIRONMENT -ErrorAction SilentlyContinue
    }
    $env:SOLACE_E2E_LANE = $Lane
    if ($RunDir) {
        $env:SOLACE_E2E_RUN_DIR = $RunDir
    } else {
        Remove-Item Env:SOLACE_E2E_RUN_DIR -ErrorAction SilentlyContinue
    }
    $env:PYTHONPATH = "$repo;$repo\src"

    $arguments = @("-m", "solace_testing.e2e", $Action)
    if ($Service) { $arguments += $Service }
    Push-Location $repo
    try {
        & $Python @arguments
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
    }
} finally {
    foreach ($name in $names) {
        [Environment]::SetEnvironmentVariable($name, $saved[$name], "Process")
    }
}
exit $code
