from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_all_seasons
from src.model import train_and_compare


def main():
    matches = load_all_seasons()
    metrics = train_and_compare(matches)
    print("Training complete")
    print("Best model:", metrics["best_model"])
    for name, result in metrics["results"].items():
        print(f"{name}: accuracy={result['accuracy']:.3f}")


if __name__ == "__main__":
    main()
