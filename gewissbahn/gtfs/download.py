from __future__ import annotations

import zipfile
from pathlib import Path

import requests

from .. import config

FEED_URL = "https://download.gtfs.de/germany/free/latest.zip"
ZIP_PATH = config.GTFS_DIR / "latest.zip"
EXTRACT_DIR = config.GTFS_DIR / "extracted"

STATIONS_REFERENCE_URL = "https://raw.githubusercontent.com/trainline-eu/stations/master/stations.csv"
STATIONS_REFERENCE_PATH = config.GTFS_DIR / "stations_reference.csv"


def download(force: bool = False) -> Path:
    config.GTFS_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and not force:
        return ZIP_PATH
    resp = requests.get(FEED_URL, stream=True, timeout=120)
    resp.raise_for_status()
    with open(ZIP_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    return ZIP_PATH


def extract(force: bool = False) -> Path:
    if EXTRACT_DIR.exists() and not force:
        return EXTRACT_DIR
    zip_path = download(force=force)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(EXTRACT_DIR)
    return EXTRACT_DIR


def download_stations_reference(force: bool = False) -> Path:
    """trainline-eu/stations: public station list with db_id (== DB eva) + lat/lon, used as
    a geo fallback when GTFS station names don't textually match piebro/DB station names."""
    config.GTFS_DIR.mkdir(parents=True, exist_ok=True)
    if STATIONS_REFERENCE_PATH.exists() and not force:
        return STATIONS_REFERENCE_PATH
    resp = requests.get(STATIONS_REFERENCE_URL, timeout=120)
    resp.raise_for_status()
    STATIONS_REFERENCE_PATH.write_bytes(resp.content)
    return STATIONS_REFERENCE_PATH
