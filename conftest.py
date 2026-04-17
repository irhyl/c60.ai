"""
conftest.py — root pytest configuration for c60.ai

Adds the src/ directory to sys.path so that `import c60` resolves
correctly when running tests without installing the package first.
"""

import sys
from pathlib import Path

# Insert src/ at the front of the path so c60.* imports resolve
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Force matplotlib to use the non-interactive Agg backend during tests
# so visualizer tests don't require a display (Tk/Qt) to be available.
try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass
