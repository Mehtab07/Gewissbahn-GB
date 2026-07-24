from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import pandas as pd

from . import download

DAY_COLUMNS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
GTFS_TABLES = ["stops", "routes", "trips", "calendar", "calendar_dates", "stop_times"]


def _parquet_cache_dir() -> Path:
    extract_dir = download.extract()
    cache_dir = extract_dir.parent / "parquet_cache"
    cache_dir.mkdir(exist_ok=True)
    con = duckdb.connect()
    for name in GTFS_TABLES:
        pq_path = cache_dir / f"{name}.parquet"
        if pq_path.exists():
            continue
        csv_path = extract_dir / f"{name}.txt"
        con.execute(
            f"COPY (SELECT * FROM read_csv_auto('{csv_path}', ignore_errors=true)) "
            f"TO '{pq_path}' (FORMAT PARQUET)"
        )
    con.close()
    return cache_dir


def connect() -> duckdb.DuckDBPyConnection:
    cache_dir = _parquet_cache_dir()
    con = duckdb.connect()
    for name in GTFS_TABLES:
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{cache_dir / (name + '.parquet')}')")
    return con


def active_service_ids(con: duckdb.DuckDBPyConnection, service_date: dt.date) -> set[str]:
    day_col = DAY_COLUMNS[service_date.weekday()]
    date_int = int(service_date.strftime("%Y%m%d"))
    base = con.execute(
        f"SELECT service_id FROM calendar WHERE start_date <= ? AND end_date >= ? AND {day_col} = 1",
        [date_int, date_int],
    ).df()["service_id"].tolist()
    removed = con.execute(
        "SELECT service_id FROM calendar_dates WHERE date = ? AND exception_type = 2", [date_int]
    ).df()["service_id"].tolist()
    added = con.execute(
        "SELECT service_id FROM calendar_dates WHERE date = ? AND exception_type = 1", [date_int]
    ).df()["service_id"].tolist()
    return (set(base) - set(removed)) | set(added)


def connections_for_date(con: duckdb.DuckDBPyConnection, service_date: dt.date) -> pd.DataFrame:
    """Time-sorted (dep_stop, dep_time, arr_stop, arr_time, trip_id) connections for CSA."""
    service_ids = list(active_service_ids(con, service_date))
    empty = pd.DataFrame(
        columns=["trip_id", "route_id", "from_stop_id", "to_stop_id", "dep_time", "arr_time", "from_seq", "to_seq"]
    )
    if not service_ids:
        return empty

    con.register("_active_service_ids", pd.DataFrame({"service_id": service_ids}))
    query = """
        WITH active_trips AS (
            SELECT trip_id, route_id FROM trips
            WHERE service_id IN (SELECT service_id FROM _active_service_ids)
        ),
        st AS (
            SELECT
                st.trip_id,
                atr.route_id,
                st.stop_id,
                st.stop_sequence,
                CAST(split_part(st.departure_time, ':', 1) AS INTEGER) * 3600
                    + CAST(split_part(st.departure_time, ':', 2) AS INTEGER) * 60
                    + CAST(split_part(st.departure_time, ':', 3) AS INTEGER) AS dep_secs,
                CAST(split_part(st.arrival_time, ':', 1) AS INTEGER) * 3600
                    + CAST(split_part(st.arrival_time, ':', 2) AS INTEGER) * 60
                    + CAST(split_part(st.arrival_time, ':', 3) AS INTEGER) AS arr_secs
            FROM stop_times st
            JOIN active_trips atr USING (trip_id)
        )
        SELECT
            trip_id,
            route_id,
            stop_id AS from_stop_id,
            LEAD(stop_id) OVER w AS to_stop_id,
            dep_secs AS dep_time,
            LEAD(arr_secs) OVER w AS arr_time,
            stop_sequence AS from_seq,
            LEAD(stop_sequence) OVER w AS to_seq
        FROM st
        WINDOW w AS (PARTITION BY trip_id ORDER BY stop_sequence)
        QUALIFY to_stop_id IS NOT NULL
        ORDER BY dep_time
    """
    result = con.execute(query).df()
    con.unregister("_active_service_ids")
    return result
