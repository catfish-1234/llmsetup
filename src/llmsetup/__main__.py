"""Allow `python -m llmsetup` alongside the installed console command."""

import sys

from llmsetup.cli import main

if __name__ == "__main__":
    sys.exit(main())
