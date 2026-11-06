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
# only ready-made builds: uv then picks the newest version of each package
# that has one for this system's glibc, instead of compiling it here
uv pip install --only-binary :all: numpy setuptools
# iisignature has no ready-made build, so it is compiled
uv pip install --no-build-isolation iisignature
# torch held at 2.5.1: later versions need glibc 2.28 (2.6's label is unclear)
uv pip install --only-binary :all: "torch==2.5.1" matplotlib pytest
.venv/bin/python -c "import torch, numpy, iisignature, matplotlib; print('setup ok: torch', torch.__version__, 'numpy', numpy.__version__)"
