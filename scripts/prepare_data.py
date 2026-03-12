import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DEFAULT_CUSTOMER_SAMPLE_FRAC
from src.pipeline import prepare_data_assets


if __name__ == "__main__":
    prepare_data_assets(sample_frac=DEFAULT_CUSTOMER_SAMPLE_FRAC)
