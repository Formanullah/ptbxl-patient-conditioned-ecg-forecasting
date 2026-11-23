#!/bin/bash
# Phase 1 evidence: environment-manager timing (uv, pip, Poetry). Linux; conda was timed
# separately via a user-local Miniforge install (see reports/phase1_foundation/REPORT.md).
# Same dependency set, cold caches, Python 3.12. Times include resolution + download + install.
set -u
B=${BENCH_DIR:-/tmp/p1bench}; mkdir -p $B; PY=/opt/python-3.12.12-1/bin/python3.12
DEPS="numpy scipy pandas matplotlib wfdb PyYAML"
rm -rf $B/uv $B/uvc $B/pip $B/pipc $B/poetry $B/poc $B/uvlock
t() { local s=$(date +%s.%N); "$@" > $B/last.log 2>&1; local rc=$?; printf "%-28s rc=%s %6.1fs\n" "$LABEL" $rc $(echo "$(date +%s.%N) - $s" | bc); [ $rc -ne 0 ] && tail -3 $B/last.log; }
export UV_CACHE_DIR=$B/uvc
LABEL="uv venv+install cold" t bash -c "uv venv -q -p $PY $B/uv && uv pip install -q --system-certs -p $B/uv/bin/python $DEPS"
rm -rf $B/uv; LABEL="uv venv+install warm" t bash -c "uv venv -q -p $PY $B/uv && uv pip install -q --system-certs -p $B/uv/bin/python $DEPS"
# uv project lock (universal, cross-platform resolution)
mkdir -p $B/uvlock && cd $B/uvlock && printf '[project]\nname="b"\nversion="0"\nrequires-python=">=3.11"\ndependencies=["numpy","scipy","pandas","matplotlib","wfdb","PyYAML"]\n' > pyproject.toml
LABEL="uv lock (universal)" t uv lock --system-certs
LABEL="uv sync from lock (warm)" t uv sync --system-certs
cd $B
LABEL="pip venv+install cold" t bash -c "$PY -m venv $B/pip && $B/pip/bin/pip install -q --cache-dir $B/pipc $DEPS"
rm -rf $B/pip; LABEL="pip venv+install warm" t bash -c "$PY -m venv $B/pip && $B/pip/bin/pip install -q --cache-dir $B/pipc $DEPS"
# poetry (installed as isolated tool via uv; install time not counted)
uv tool install -q --system-certs poetry > /dev/null 2>&1; POETRY=$(uv tool dir --bin)/poetry
mkdir -p $B/poetry && cd $B/poetry && printf '[project]\nname="b"\nversion="0"\nrequires-python=">=3.11,<3.15"\ndependencies=["numpy","scipy","pandas","matplotlib","wfdb","PyYAML"]\n[tool.poetry]\npackage-mode=false\n' > pyproject.toml
export POETRY_CACHE_DIR=$B/poc POETRY_VIRTUALENVS_IN_PROJECT=true
$POETRY env use $PY > /dev/null 2>&1
LABEL="poetry lock cold" t $POETRY lock
LABEL="poetry install cold" t $POETRY install --no-root
$POETRY --version
echo "uv env size: $(du -sh $B/uv | cut -f1), pip env size: $(du -sh $B/pip | cut -f1)"
