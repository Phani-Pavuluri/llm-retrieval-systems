import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import load_reviews


def main() -> None:
    df = load_reviews()
    print(df.head())
    print("\nShape:", df.shape)
    print("\nColumns:", df.columns.tolist())


if __name__ == "__main__":
    main()