# Starts everything, each in its own window: Kev, the chat model, the cloned voice, then m8 once
# they're up (it waits for them), and the game if one is built here.
function Test-Port($port) {
    try { Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 1 | Out-Null; $true } catch { $false }
}

if (-not (Test-Port 8080)) { Start-Process pwsh -ArgumentList '-NoExit', '-File', "$PSScriptRoot\start-kev.ps1" }
if (-not (Test-Port 8081)) { Start-Process pwsh -ArgumentList '-NoExit', '-File', "$PSScriptRoot\start-chat.ps1" }
if (-not (Test-Port 7860)) { Start-Process pwsh -ArgumentList '-NoExit', '-File', "$PSScriptRoot\start-breeze.ps1" }

$m8 = "while (-not ((Invoke-WebRequest http://127.0.0.1:8080/health -SkipHttpErrorCheck -TimeoutSec 1 -ErrorAction SilentlyContinue).StatusCode -eq 200 -and " +
    "(Invoke-WebRequest http://127.0.0.1:8081/health -SkipHttpErrorCheck -TimeoutSec 1 -ErrorAction SilentlyContinue).StatusCode -eq 200 -and " +
    "(Invoke-WebRequest http://127.0.0.1:7860/health -SkipHttpErrorCheck -TimeoutSec 1 -ErrorAction SilentlyContinue).StatusCode -eq 200)) { Start-Sleep 2 }; " +
    "& '$PSScriptRoot\m8\target\release\m8.exe' m8.toml"
Start-Process pwsh -WorkingDirectory $PSScriptRoot -ArgumentList '-NoExit', '-Command', $m8

# (the game, if there's one here: it connects to m8 itself, again whenever m8 restarts)
$game = "$PSScriptRoot\halo\build\windows\halo.exe"
if (Test-Path $game) { Start-Process $game -WorkingDirectory (Split-Path $game) }
