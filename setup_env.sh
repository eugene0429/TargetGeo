#!/usr/bin/env bash
# Self-contained dev environment setup for TargetGeo (targetgeo).
# Creates a local .venv, installs all deps, and wires the package import name —
# fully independent of any other project/sandbox on the machine.
#
# Usage:  ./setup_env.sh
# Re-run safe: recreates .venv from scratch.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_DIR/.venv"
PYBIN="${PYTHON:-python3.10}"
TORCH_INDEX="https://download.pytorch.org/whl/cu126"

echo "==> repo:  $REPO_DIR"
echo "==> venv:  $VENV"
echo "==> python: $("$PYBIN" --version)"

# 1) Fresh venv
rm -rf "$VENV"
"$PYBIN" -m venv "$VENV"
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"
"$PIP" install --upgrade pip wheel setuptools

# 2) torch (CUDA 12.6 build, proven on this RTX 3090 box) + torchvision
"$PIP" install --index-url "$TORCH_INDEX" torch torchvision

# 3) Remaining deps (numpy gets pinned <2 by sam3's constraint).
#    requirements.txt includes "sam3 @ git+https://github.com/facebookresearch/sam3".
"$PIP" install -r "$REPO_DIR/requirements.txt"

# 4) Make the package importable as `targetgeo` INSIDE this venv only
#    (no global symlinks; matches the import name used by the test suite).
SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
ln -sfn "$REPO_DIR" "$SITE/targetgeo"
echo "==> linked $SITE/targetgeo -> $REPO_DIR"

# 5) Detector weights: replace the drone symlink with a real in-repo copy
#    (gitignored per README — weights are supplied per deployment).
WEIGHTS="$REPO_DIR/models/target_detector.pt"
if [ -L "$WEIGHTS" ]; then
  TARGET="$(readlink -f "$WEIGHTS")"
  if [ -f "$TARGET" ]; then
    cp --remove-destination "$TARGET" "$WEIGHTS"
    echo "==> copied real detector weights into repo ($(du -h "$WEIGHTS" | cut -f1))"
  else
    echo "!! WARNING: detector weights symlink target missing: $TARGET" >&2
  fi
fi

# 6) Smoke check — run from a neutral cwd so the repo's own sam3.py does not
#    shadow the installed top-level `sam3` package (cwd is on sys.path).
echo "==> verifying environment"
cd /tmp
"$PY" - <<'PYEOF'
import torch, cv2, numpy as np
import sam3.model_builder  # noqa: F401
import targetgeo
from targetgeo import TargetGeoEstimator  # noqa: F401
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("numpy", np.__version__, "cv2", cv2.__version__)
print("targetgeo ->", targetgeo.__file__)
print("OK")
PYEOF

echo ""
echo "==> done. Use: $VENV/bin/python"
echo "    tests:  $VENV/bin/python -m pytest -m 'not slow' -q"
