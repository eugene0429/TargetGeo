# Launch the interactive viewer in the repo-local venv — Windows port of run_viewer.sh.
# Usage:  .\run_viewer.ps1 <video.mp4 | rtsp://host/stream> [viewer args...]

$ErrorActionPreference = "Stop"

$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$PY      = Join-Path $RepoDir ".venv\Scripts\python.exe"

if (-not (Test-Path $PY)) {
    Write-Error "venv missing - run .\setup_env.ps1 first"
    exit 1
}

& $PY -m targetgeo.viewer @args
exit $LASTEXITCODE
