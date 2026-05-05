from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_SAMPLE_FILE = RAW_DATA_DIR / "quito_air_quality_sample.csv"
PROCESSED_SAMPLE_FILE = PROCESSED_DATA_DIR / "quito_air_quality_processed.csv"

REAL_RAW_FILE = RAW_DATA_DIR / "quito_air_quality_real.csv"
REAL_PROCESSED_FILE = PROCESSED_DATA_DIR / "quito_air_quality_real_processed.csv"