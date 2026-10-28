"""Allow executing ChurnLens via `python -m churnlens`."""

from __future__ import annotations

import sys

from churnlens.main import main

if __name__ == "__main__":
    sys.exit(main())
