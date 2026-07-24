import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
GTFS_DIR = DATA_DIR / "gtfs"
HISTORICAL_DATA_GLOB = str(DATA_DIR / "monthly_processed_data" / "*.parquet")

DB_CLIENT_ID = os.environ["Client_ID"]
DB_API_KEY = os.environ["Client_API"]
TIMETABLES_BASE_URL = "https://apis.deutschebahn.com/db-api-marketplace/apis/timetables/v1"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
REASONING_MODEL = os.environ.get("REASONING_MODEL", "openai/gpt-4o-mini")
