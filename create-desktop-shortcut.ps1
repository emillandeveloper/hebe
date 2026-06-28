$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$launcherPath = Join-Path -Path $projectRoot -ChildPath "start-hebe.bat"

if (-not (Test-Path -LiteralPath $launcherPath)) {
    Write-Host "ERROR: Could not find launcher: `"$launcherPath`"" -ForegroundColor Red
    exit 1
}

$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path -Path $desktopPath -ChildPath "Hebe.lnk"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $launcherPath
$shortcut.WorkingDirectory = $projectRoot
$shortcut.WindowStyle = 1
$shortcut.Description = "Start Hebe in development mode"

$icon = Get-ChildItem -LiteralPath $projectRoot -Recurse -Filter "*.ico" -File -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($icon) {
    $shortcut.IconLocation = $icon.FullName
}

$shortcut.Save()

Write-Host "Created desktop shortcut:"
Write-Host "`"$shortcutPath`""
Write-Host ""
Write-Host "It points to:"
Write-Host "`"$launcherPath`""
