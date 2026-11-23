#!/bin/sh
# Download and verify PTB-XL v1.0.3 (Linux/macOS). No admin rights needed.
# Uses uv if installed, otherwise python3 (3.9+). All arguments are passed through:
#   sh scripts/download_ptbxl.sh --root PTB-XL
#   sh scripts/download_ptbxl.sh --root PTB-XL --verify-only
set -e
here=$(cd "$(dirname "$0")" && pwd)
if command -v uv >/dev/null 2>&1; then
    exec uv run --no-project python "$here/download_ptbxl.py" "$@"
elif command -v python3 >/dev/null 2>&1; then
    exec python3 "$here/download_ptbxl.py" "$@"
else
    echo "Install uv (https://docs.astral.sh/uv/) or Python 3.9+ first." >&2
    exit 1
fi
