"""Allow `python -m llmfit` alongside the installed console command."""

import sys

from llmfit.cli import main

if __name__ == "__main__":
    sys.exit(main())
