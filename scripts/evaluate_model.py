"""Evaluate the saved model on the held-out test split; writes reports/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
