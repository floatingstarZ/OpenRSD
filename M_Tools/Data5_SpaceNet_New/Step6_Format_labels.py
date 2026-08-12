#!/usr/bin/env python3
"""Compatibility wrapper for the portable OpenRSD Step6 formatter."""

import runpy
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "OpenRSD_Reproduction" / "format_training_labels.py"
runpy.run_path(str(SCRIPT), run_name="__main__")

