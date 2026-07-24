from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import duckdb
import pandas as pd

from . import config
from .gtfs.station_mapping import eva_for_platform
from .routing.csa import Itinerary

MIN_SAMPLE_SIZE = 20
SEASON_MONTHS = {
    "winter": {12, 1, 2},
    "spring": {3, 4, 5},
    "summer": {6, 7, 8},
    "autumn": {9, 10, 11},
}


@dataclass
class TransferEstimate:
    eva: str | None
    success_probability: float
    sample_size: int
    specificity: str


@dataclass
class ItineraryScore:
    transfers: list[TransferEstimate]

    @property
    def confidence(self) -> float:
        """Overall probability of making every transfer. Treats transfers as independent
        (a simplification — real transfers at the same hub on the same day can be
        correlated, e.g. a signal fault delaying several lines at once — a reasonable
        first pass, revisit if the trained model in a later phase needs to account for it)."""
        p = 1.0
        for t in self.transfers:
            p *= t.success_probability
        return p


def _season_for_month(month: int) -> str:
    for season, months in SEASON_MONTHS.items():
        if month in months:
            return season
    raise ValueError(month)


def _build_query(
    eva: str,
    buffer_min: float,
    hour: int,
    hour_window: int,
    weekday: int | None,
    season: str | None,
    train_type: str | None,
) -> tuple[str, list]:
    where = ["eva = ?", "NOT is_canceled", "arrival_planned_time IS NOT NULL", "arrival_change_time IS NOT NULL"]
    where_params: list = [eva]

    lo, hi = max(0, hour - hour_window), min(23, hour + hour_window)
    where.append("extract(hour from arrival_planned_time) BETWEEN ? AND ?")
    where_params += [lo, hi]

    if weekday is not None:
        if weekday >= 5:
            where.append("dayofweek(arrival_planned_time) IN (0, 6)")  # DuckDB: Sun=0..Sat=6
        else:
            where.append("dayofweek(arrival_planned_time) BETWEEN 1 AND 5")

    if season is not None:
        months = ",".join(str(m) for m in SEASON_MONTHS[season])
        where.append(f"extract(month from arrival_planned_time) IN ({months})")

    if train_type is not None:
        where.append("train_type = ?")
        where_params.append(train_type)

    sql = f"""
        SELECT
            count(*) AS n,
            avg(CASE WHEN date_diff('minute', arrival_planned_time, arrival_change_time) <= ?
                THEN 1.0 ELSE 0.0 END) AS success_rate
        FROM read_parquet('{config.HISTORICAL_DATA_GLOB}')
        WHERE {" AND ".join(where)}
    """
    return sql, [buffer_min] + where_params


def transfer_success_probability(
    con: duckdb.DuckDBPyConnection,
    eva: str,
    when: dt.datetime,
    buffer_min: float,
    train_type: str | None = None,
) -> TransferEstimate:
    """Empirical P(arrival delay <= scheduled buffer) for arrivals at this station under
    similar conditions. Backs off to broader conditions when a specific combination has
    too few historical samples to trust."""
    hour, weekday, season = when.hour, when.weekday(), _season_for_month(when.month)

    levels = []
    if train_type:
        levels += [
            {"hour_window": 1, "weekday": weekday, "season": season, "train_type": train_type},
            {"hour_window": 2, "weekday": weekday, "season": season, "train_type": train_type},
        ]
    levels += [
        {"hour_window": 1, "weekday": weekday, "season": season, "train_type": None},
        {"hour_window": 2, "weekday": weekday, "season": season, "train_type": None},
        {"hour_window": 2, "weekday": weekday, "season": None, "train_type": None},
        {"hour_window": 3, "weekday": None, "season": None, "train_type": None},
        {"hour_window": 12, "weekday": None, "season": None, "train_type": None},
    ]

    for level in levels:
        sql, params = _build_query(eva, buffer_min, hour, **level)
        n, success_rate = con.execute(sql, params).fetchone()
        if n and n >= MIN_SAMPLE_SIZE:
            tt = level["train_type"] or "any"
            wd = "weekday" if level["weekday"] == 0 else ("weekend" if level["weekday"] == 5 else "any-day") if level["weekday"] is not None else "any-day"
            label = f"train_type={tt}, hour±{level['hour_window']}, day={wd}, season={level['season'] or 'any'}, n={n}"
            return TransferEstimate(eva=eva, success_probability=success_rate, sample_size=n, specificity=label)

    return TransferEstimate(eva=eva, success_probability=0.5, sample_size=0, specificity="no_historical_data")


def score_itinerary(
    hist_con: duckdb.DuckDBPyConnection,
    gtfs_con: duckdb.DuckDBPyConnection,
    mapping: pd.DataFrame,
    itinerary: Itinerary,
    service_date: dt.date,
) -> ItineraryScore:
    transfers = []
    buffers = itinerary.transfer_buffers()
    for prev_leg, next_leg, buffer_sec in zip(itinerary.legs, itinerary.legs[1:], buffers):
        if buffer_sec <= 0 and prev_leg.alight_stop_id == next_leg.board_stop_id:
            # same platform, zero buffer: a through-service continuation, not a real
            # transfer that could be missed (see PROGRESS.md Step 3 note)
            transfers.append(
                TransferEstimate(eva=None, success_probability=1.0, sample_size=0, specificity="same_vehicle")
            )
            continue

        eva = eva_for_platform(gtfs_con, mapping, prev_leg.alight_stop_id)
        arrival_dt = dt.datetime.combine(service_date, dt.time(0, 0)) + dt.timedelta(seconds=prev_leg.alight_time)
        if eva is None:
            transfers.append(
                TransferEstimate(eva=None, success_probability=0.5, sample_size=0, specificity="unmapped_station")
            )
            continue

        estimate = transfer_success_probability(hist_con, eva, arrival_dt, buffer_sec / 60)
        transfers.append(estimate)

    return ItineraryScore(transfers=transfers)
