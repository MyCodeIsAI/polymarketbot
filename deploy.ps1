#
# PolymarketBot - Windows Deployment Script
#
# Run this script in PowerShell to deploy PolymarketBot on Windows.
#
# Usage:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#   .\deploy.ps1
#

param(
    [int]$Port = 8765,
    [string]$InstallDir = "$env:USERPROFILE\polymarketbot",
    [switch]$UseDocker = $false
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "           PolymarketBot Deployer (Windows)             " -ForegroundColor Cyan
Write-Host "          Low-Latency Copy Trading System               " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
function Test-Python {
    try {
        $version = python --version 2>&1
        if ($version -match "Python 3\.(\d+)") {
            $minor = [int]$Matches[1]
            if ($minor -ge 10) {
                Write-Host "Found: $version" -ForegroundColor Green
                return $true
            }
        }
    } catch {}

    Write-Host "Python 3.10+ is required." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    return $false
}

# Install Python dependencies
function Install-PythonDeps {
    Write-Host "Setting up Python environment..." -ForegroundColor Cyan

    # Create virtual environment
    python -m venv venv

    # Activate and install dependencies
    & .\venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt

    Write-Host "Python environment ready." -ForegroundColor Green
}

# Create Windows Service using NSSM (if available)
function Install-WindowsService {
    $nssm = Get-Command nssm -ErrorAction SilentlyContinue

    if ($nssm) {
        Write-Host "Creating Windows service with NSSM..." -ForegroundColor Cyan

        nssm install PolymarketBot "$InstallDir\venv\Scripts\python.exe"
        nssm set PolymarketBot AppParameters "run_ghost_mode.py"
        nssm set PolymarketBot AppDirectory "$InstallDir"
        nssm set PolymarketBot AppEnvironmentExtra "PORT=$Port"
        nssm set PolymarketBot Start SERVICE_AUTO_START

        Write-Host "Service created. Start with: nssm start PolymarketBot" -ForegroundColor Green
    } else {
        Write-Host "NSSM not found. Install from https://nssm.cc for Windows service support." -ForegroundColor Yellow
    }
}

# Create startup script
function Create-StartupScript {
    $startScript = @"
@echo off
cd /d "$InstallDir"
call venv\Scripts\activate.bat
python run_ghost_mode.py --port $Port
pause
"@

    $startScript | Out-File -FilePath "$InstallDir\start.bat" -Encoding ASCII
    Write-Host "Created start.bat" -ForegroundColor Green

    # Create shortcut on desktop
    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\PolymarketBot.lnk")
    $Shortcut.TargetPath = "$InstallDir\start.bat"
    $Shortcut.WorkingDirectory = $InstallDir
    $Shortcut.Save()

    Write-Host "Created desktop shortcut" -ForegroundColor Green
}

# Main deployment
function Main {
    Write-Host "Install directory: $InstallDir" -ForegroundColor Cyan
    Write-Host "Dashboard port: $Port" -ForegroundColor Cyan
    Write-Host ""

    # Check Python
    if (-not (Test-Python)) {
        exit 1
    }

    # Setup directory
    if (-not (Test-Path $InstallDir)) {
        Write-Host "Creating install directory..." -ForegroundColor Cyan
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    }

    # Check if we're in the repo or need to clone
    $currentDir = Get-Location
    if (Test-Path "run_ghost_mode.py") {
        Write-Host "Using current directory as source..." -ForegroundColor Cyan
        if ($currentDir.Path -ne $InstallDir) {
            Copy-Item -Path ".\*" -Destination $InstallDir -Recurse -Force
        }
    } else {
        Write-Host "Please clone the repository first:" -ForegroundColor Yellow
        Write-Host "  git clone https://github.com/yourusername/polymarketbot.git $InstallDir" -ForegroundColor Yellow
        exit 1
    }

    Set-Location $InstallDir

    # Install Python dependencies
    Install-PythonDeps

    # Create startup scripts
    Create-StartupScript

    # Try to create Windows service
    Install-WindowsService

    Write-Host ""
    Write-Host "========================================================" -ForegroundColor Green
    Write-Host "              Deployment Complete!                      " -ForegroundColor Green
    Write-Host "========================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Dashboard URL: http://localhost:$Port" -ForegroundColor Cyan
    Write-Host "Infrastructure: http://localhost:$Port/infrastructure" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Start the bot:" -ForegroundColor Yellow
    Write-Host "  Double-click 'PolymarketBot' shortcut on desktop" -ForegroundColor White
    Write-Host "  Or run: .\start.bat" -ForegroundColor White
    Write-Host ""
    Write-Host "Manual start:" -ForegroundColor Yellow
    Write-Host "  cd $InstallDir" -ForegroundColor White
    Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host "  python run_ghost_mode.py" -ForegroundColor White
    Write-Host ""
}

# Run main function
Main
