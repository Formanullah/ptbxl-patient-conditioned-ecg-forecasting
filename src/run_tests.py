<<<<<<< HEAD
"""Run software checks with: python src/run_tests.py -q."""
=======
"""Run the active scientific and software checks with pytest."""
>>>>>>> origin/work/manual-src-workflow

import sys

import pytest

<<<<<<< HEAD
raise SystemExit(pytest.main(sys.argv[1:]))
=======
if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:]))
>>>>>>> origin/work/manual-src-workflow
