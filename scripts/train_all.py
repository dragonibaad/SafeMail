import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from safemail.train_classical import train_and_save_classical_models
from safemail.train_distilbert import train_and_save_distilbert


def main() -> None:
    train_and_save_classical_models()
    train_and_save_distilbert()


if __name__ == "__main__":
    main()
