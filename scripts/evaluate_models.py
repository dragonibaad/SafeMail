import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from safemail.evaluate import evaluate_and_export


def main() -> None:
    report = evaluate_and_export()
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
