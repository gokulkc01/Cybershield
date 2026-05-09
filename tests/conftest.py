import sys
from pathlib import Path

# Ensure repository root is on sys.path so `src` imports work under pytest
ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
