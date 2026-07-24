from __future__ import annotations

import difflib
import re
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd

from .. import config
from . import download

_STRIP_PREFIX_RE = re.compile(r"^(s\+u|s\+bus|s|u|bus|tram)\s+", re.IGNORECASE)
_HAUPTBAHNHOF_RE = re.compile(r"hauptbahnhof", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")

EARTH_RADIUS_M = 6_371_000
GEO_MATCH_RADIUS_M = 400


def normalize_name(name: str) -> str:
    name = name.replace(",", " ")
    name = _STRIP_PREFIX_RE.sub("", name)
    name = _HAUPTBAHNHOF_RE.sub("hbf", name)
    name = name.lower()
    words = name.split()
    seen = set()
    deduped = []
    for w in words:
        if w not in seen:
            seen.add(w)
            deduped.append(w)
    name = " ".join(deduped)
    return _NON_ALNUM_RE.sub("", name)


def piebro_stations() -> pd.DataFrame:
    """Distinct eva -> most common xml_station_name across the historical dataset."""
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT eva, xml_station_name AS name, count(*) AS n
        FROM read_parquet('{config.HISTORICAL_DATA_GLOB}')
        WHERE xml_station_name IS NOT NULL
        GROUP BY eva, xml_station_name
    """).df()
    con.close()
    df = df.sort_values("n", ascending=False).drop_duplicates("eva")
    return df[["eva", "name"]].reset_index(drop=True)


def gtfs_stations(gtfs_con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """All GTFS stops usable as match candidates: proper stations (location_type=1) plus
    any stop that itself has no parent (some real stations aren't tagged location_type=1
    in this feed, e.g. Stuttgart's main hub is a plain stop). Some station names have
    duplicate near-empty entries in this feed (e.g. a second, childless "Koeln Hbf" next
    to the real one, "Koeln Breslauer Platz/Hbf", which has 12 platforms) — traffic
    (total stop_times across all child platforms) is attached so callers can disambiguate."""
    return gtfs_con.execute("""
        WITH candidates AS (
            SELECT stop_id, stop_name, stop_lat, stop_lon
            FROM stops
            WHERE location_type = 1 OR parent_station IS NULL
        ),
        child_traffic AS (
            SELECT s.parent_station AS station_id, count(*) AS n
            FROM stops s
            JOIN stop_times st ON st.stop_id = s.stop_id
            WHERE s.parent_station IS NOT NULL
            GROUP BY s.parent_station
        )
        SELECT
            c.stop_id, c.stop_name AS name, c.stop_lat, c.stop_lon,
            coalesce(t.n, 0) AS traffic
        FROM candidates c
        LEFT JOIN child_traffic t ON t.station_id = c.stop_id
    """).df()


def stations_reference() -> pd.DataFrame:
    """eva -> lat/lon from trainline-eu/stations, used for geo fallback matching."""
    path = download.download_stations_reference()
    df = pd.read_csv(path, sep=";", usecols=["db_id", "latitude", "longitude"], dtype={"db_id": "string"})
    df = df.dropna(subset=["db_id", "latitude", "longitude"])
    df["db_id"] = df["db_id"].str.lstrip("0")
    return df.drop_duplicates("db_id").set_index("db_id")


def _haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def build_mapping(gtfs_con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """eva <-> GTFS stop_id, matched by normalized station name, then geo distance."""
    piebro = piebro_stations()
    gtfs = gtfs_stations(gtfs_con)

    piebro["norm"] = piebro["name"].apply(normalize_name)
    gtfs["norm"] = gtfs["name"].apply(normalize_name)
    gtfs = gtfs.sort_values("traffic", ascending=False)

    gtfs_by_norm: dict[str, list[str]] = defaultdict(list)
    for _, row in gtfs.iterrows():
        gtfs_by_norm[row["norm"]].append(row["stop_id"])  # highest-traffic first, per sort above

    buckets: dict[str, list[str]] = defaultdict(list)
    for norm in gtfs_by_norm:
        buckets[norm[:3]].append(norm)

    matches = []
    for _, row in piebro.iterrows():
        norm = row["norm"]
        candidates = gtfs_by_norm.get(norm)
        method = "exact"
        if not candidates:
            bucket_choices = buckets.get(norm[:3], [])
            close = difflib.get_close_matches(norm, bucket_choices, n=1, cutoff=0.85)
            if close:
                candidates = gtfs_by_norm[close[0]]
                method = "fuzzy"
        matches.append({
            "eva": row["eva"],
            "station_name": row["name"],
            "gtfs_stop_id": candidates[0] if candidates else None,
            "match_method": method if candidates else "unmatched",
        })
    result = pd.DataFrame(matches)

    # geo fallback for anything still unmatched — only consider candidates with real traffic,
    # otherwise a near-empty duplicate entry can win purely on proximity
    unmatched_mask = result["match_method"] == "unmatched"
    if unmatched_mask.any():
        ref = stations_reference()
        geo_candidates = gtfs[gtfs["traffic"] > 0]
        gtfs_lat = geo_candidates["stop_lat"].to_numpy(dtype=float)
        gtfs_lon = geo_candidates["stop_lon"].to_numpy(dtype=float)
        gtfs_ids = geo_candidates["stop_id"].to_numpy()

        for idx in result[unmatched_mask].index:
            eva = result.at[idx, "eva"]
            key = eva.lstrip("0")
            if key not in ref.index:
                continue
            row = ref.loc[key]
            lat, lon = row["latitude"], row["longitude"]
            if isinstance(lat, pd.Series):  # duplicate index safety
                lat, lon = lat.iloc[0], lon.iloc[0]
            dists = _haversine_m(lat, lon, gtfs_lat, gtfs_lon)
            best = np.argmin(dists)
            if dists[best] <= GEO_MATCH_RADIUS_M:
                result.at[idx, "gtfs_stop_id"] = gtfs_ids[best]
                result.at[idx, "match_method"] = "geo"

    return result


def get_mapping(gtfs_con: duckdb.DuckDBPyConnection, force: bool = False) -> pd.DataFrame:
    cache_path = config.GTFS_DIR / "station_mapping.parquet"
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)
    mapping = build_mapping(gtfs_con)
    mapping.to_parquet(cache_path)
    return mapping


def primary_eva_for_stop(mapping: pd.DataFrame, gtfs_stop_id: float) -> str | None:
    """Several piebro evas can map to the same physical GTFS station (e.g. a station's
    long-distance eva and its separate S-Bahn-only eva both resolve to one GTFS entity).
    Picks the most likely "main" eva: prefer a name with no parenthetical/suffix qualifier
    (avoid "Berlin Hbf (S-Bahn)" in favor of plain "Berlin Hbf"), then the shortest name."""
    candidates = mapping[mapping["gtfs_stop_id"] == gtfs_stop_id]
    if candidates.empty:
        return None

    def rank(name: str) -> tuple[int, int]:
        has_qualifier = 1 if "(" in name else 0
        return (has_qualifier, len(name))

    candidates = candidates.assign(_rank=candidates["station_name"].apply(rank))
    return candidates.sort_values("_rank").iloc[0]["eva"]


def eva_for_platform(gtfs_con: duckdb.DuckDBPyConnection, mapping: pd.DataFrame, platform_stop_id: int) -> str | None:
    """A leg's board/alight stop_id is a specific platform; resolve it up to its station,
    then to the most likely eva for that station (see primary_eva_for_stop)."""
    row = gtfs_con.execute(
        "SELECT coalesce(parent_station, stop_id) AS station_id FROM stops WHERE stop_id = ?", [platform_stop_id]
    ).fetchone()
    if row is None:
        return None
    return primary_eva_for_stop(mapping, float(row[0]))


def platform_stop_ids(gtfs_con: duckdb.DuckDBPyConnection, station_stop_id: str) -> list[str]:
    df = gtfs_con.execute(
        "SELECT stop_id FROM stops WHERE parent_station = ?", [station_stop_id]
    ).df()
    ids = df["stop_id"].tolist()
    return ids or [station_stop_id]
