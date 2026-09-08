"""Launcher for the handwriting recognition desktop app."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from ocr_project.app import main  # noqa: E402

if __name__ == "__main__":
    main()
