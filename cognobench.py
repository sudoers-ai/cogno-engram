#!/usr/bin/env python3
"""Root shim — run the engram bench with `python3 cognobench.py`.

Examples:
    python3 cognobench.py
    python3 cognobench.py --only retrieval --limit 3
    python3 cognobench.py --min-score 100        # CI gate
"""

import sys

from cognobench.runner import main

if __name__ == "__main__":
    sys.exit(main())
