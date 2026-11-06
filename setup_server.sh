#!/usr/bin/env bash
# one-time setup on an old linux server (debian 9: gcc 6, glibc 2.24)
#   ./setup_server.sh
# needs uv in ~/.local/bin; makes .venv in the repo folder
set -euo pipefail
cd "$(dirname "$0")"
export PATH=$HOME/.local/bin:$PATH
command -v uv > /dev/null || { echo "uv not found in ~/.local/bin"; exit 1; }

# the python uv installs asks for a compiler flag gcc 6 does not know
T=$PWD/tools
export CC=$T/cc-nosc CXX=$T/cxx-nosc
export LDSHARED="$T/cc-nosc -shared" LDCXXSHARED="$T/cxx-nosc -shared"

[ -d .venv ] || uv venv --python 3.12
# newest versions with builds for glibc 2.17, checked on pypi 2026-11-06
uv pip install "numpy<2.3" setuptools
uv pip install --no-build-isolation iisignature
uv pip install "torch==2.5.1" "matplotlib<3.11" "pillow<13" pytest
.venv/bin/python -c "import torch, numpy, iisignature, matplotlib; print('setup ok: torch', torch.__version__, 'numpy', numpy.__version__)"
