$ErrorActionPreference = "Stop"

function Pause-OnError {
    param([int]$ExitCode = 1)

    Write-Host ""
    Write-Host "Startup failed. Press any key to close." -ForegroundColor Red
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit $ExitCode
}

try {
    Write-Host ""
    Write-Host "Starting Hebe..."
    Write-Host ""

    $projectRoot = $PSScriptRoot
    $frontendDir = Join-Path -Path $projectRoot -ChildPath "frontend"

    Write-Host "Checking Node..."
    if (-not (Get-Command node.exe -ErrorAction SilentlyContinue)) {
        Write-Host "ERROR: Node.js was not found on PATH." -ForegroundColor Red
        Write-Host "Install Node.js or add it to PATH, then try again."
        Pause-OnError
    }

    Write-Host "Checking npm..."
    $npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npmCommand) {
        Write-Host "ERROR: npm was not found on PATH." -ForegroundColor Red
        Write-Host "Install Node.js/npm or add it to PATH, then try again."
        Pause-OnError
    }

    $packageJson = Join-Path -Path $frontendDir -ChildPath "package.json"
    if (-not (Test-Path -LiteralPath $packageJson)) {
        Write-Host "ERROR: Could not find frontend\package.json." -ForegroundColor Red
        Write-Host "Project root: `"$projectRoot`""
        Write-Host "Expected frontend folder: `"$frontendDir`""
        Pause-OnError
    }

    Write-Host "Changing directory to:"
    Write-Host "`"$frontendDir`""
    Set-Location -LiteralPath $frontendDir

    Write-Host ""
    Write-Host "Running electron dev:"
    Write-Host "npm run electron:dev"
    Write-Host ""

    & $npmCommand.Source run electron:dev
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Host ""
        Write-Host "Startup failed with exit code $exitCode." -ForegroundColor Red
        Pause-OnError -ExitCode $exitCode
    }

    exit $exitCode
}
catch {
    Write-Host ""
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Pause-OnError
}
