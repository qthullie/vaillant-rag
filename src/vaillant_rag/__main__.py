"""Allow running as ``python -m vaillant_rag``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
