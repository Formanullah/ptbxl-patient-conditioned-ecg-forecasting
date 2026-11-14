"""Run software checks with: python src/run_tests.py -q."""

import sys

import pytest

raise SystemExit(pytest.main(sys.argv[1:]))
