# Self-contained dev environment setup for TargetGeo (targetgeo) — Windows port of setup_env.sh.
# Creates a local .venv, installs all deps, and wires the package import name.
#
# Usage (from repo root, PowerShell):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_env.ps1
# Re-run safe: recreates .venv from scratch.
# Override interpreter with $env:PYTHON (e.g. "py -3.10" or a full python.exe path).

$ErrorActionPreference = "Stop"

$RepoDir    = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Venv       = Join-Path $RepoDir ".venv"
$TorchIndex = "https://download.pytorch.org/whl/cu126"

# Resolve the bootstrap interpreter (supports "py -3.10" style or a plain exe path).
if ($env:PYTHON) {
    $parts   = $env:PYTHON.Split(" ")
    $PyExe   = $parts[0]
    $PyArgs  = $parts[1..($parts.Length - 1)]
} else {
    $PyExe   = "py"
    $PyArgs  = @("-3.10")
}

Write-Host "==> repo:   $RepoDir"
Write-Host "==> venv:   $Venv"
Write-Host "==> python: $(& $PyExe @PyArgs --version)"

# 1) Fresh venv
if (Test-Path $Venv) { Remove-Item -Recurse -Force $Venv }
& $PyExe @PyArgs -m venv $Venv

$PIP = Join-Path $Venv "Scripts\pip.exe"
$PY  = Join-Path $Venv "Scripts\python.exe"
& $PIP install --upgrade pip wheel setuptools

# 2) torch (CUDA 12.6 build) + torchvision
& $PIP install --index-url $TorchIndex torch torchvision

# 3) Remaining deps (sam3 from git + under-declared training-stack deps).
& $PIP install -r (Join-Path $RepoDir "requirements.txt")

# 4) Make the package importable as `targetgeo` inside this venv only,
#    via a directory junction (Windows has no symlink-by-default; junctions need no admin).
$Site = & $PY -c "import site; print(site.getsitepackages()[0])"
$Link = Join-Path $Site "targetgeo"
if (Test-Path $Link) { Remove-Item -Recurse -Force $Link }
New-Item -ItemType Junction -Path $Link -Target $RepoDir | Out-Null
Write-Host "==> linked $Link -> $RepoDir"

# 5) Detector weights notice (supplied per deployment; see README / WINDOWS_SETUP.md).
$Weights = Join-Path $RepoDir "models\target_detector.pt"
if (-not (Test-Path $Weights)) {
    Write-Warning "detector weights not found at $Weights — copy them in before running (see docs\WINDOWS_SETUP.md §5)."
}

# 6) Smoke check — run from a neutral cwd so the repo's own sam3.py does not
#    shadow the installed top-level `sam3` package (cwd is on sys.path).
Write-Host "==> verifying environment"
Push-Location $env:TEMP
try {
    & $PY -c @"
import torch, cv2, numpy as np
import sam3.model_builder  # noqa: F401
import targetgeo
from targetgeo import TargetGeoEstimator  # noqa: F401
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('numpy', np.__version__, 'cv2', cv2.__version__)
print('targetgeo ->', targetgeo.__file__)
print('OK')
"@
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "==> done. Use: $PY"
Write-Host "    tests:  cd ..; $PY -m pytest TargetGeo\tests\ -m 'not slow' -q"
