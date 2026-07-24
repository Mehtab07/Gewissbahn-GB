from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import duckdb
import pandas as pd

from .. import timetables_api
from ..timetables_api import StopEvent
from ..gtfs.station_mapping import eva_for_platform, normalize_name
from .csa import Itinerary, Leg

MATCH_TOLERANCE_SECONDS = 180
_TIME_FMT = "%y%m%d%H%M"


@dataclass
class LiveLeg:
    leg: Leg
    board_eva: str | None
    alight_eva: str | None
    board_live: StopEvent | None
    alight_live: StopEvent | None

    @property
    def board_delay_min(self) -> int | None:
        return _delay_minutes(self.board_live, "departure") if self.board_live else None

    @property
    def alight_delay_min(self) -> int | None:
        return _delay_minutes(self.alight_live, "arrival") if self.alight_live else None

    @property
    def is_cancelled(self) -> bool:
        return bool(self.board_live and self.board_live.departure_cancelled) or bool(
            self.alight_live and self.alight_live.arrival_cancelled
        )


def _delay_minutes(event: StopEvent, kind: str) -> int | None:
    planned = getattr(event, f"{kind}_planned")
    changed = getattr(event, f"{kind}_changed")
    if not planned or not changed:
        return None
    try:
        return int(
            (dt.datetime.strptime(changed, _TIME_FMT) - dt.datetime.strptime(planned, _TIME_FMT)).total_seconds()
            // 60
        )
    except ValueError:
        return None


def _seconds_to_datetime(base_date: dt.date, secs: int) -> dt.datetime:
    return dt.datetime.combine(base_date, dt.time(0, 0)) + dt.timedelta(seconds=secs)


def _path_contains(path: list[str], target_name: str | None) -> bool:
    if not target_name:
        return False
    target_norm = normalize_name(target_name)
    return any(normalize_name(p) == target_norm for p in path)


def _best_match(events: list[StopEvent], target: dt.datetime, kind: str, through_name: str | None) -> StopEvent | None:
    candidates = []
    for e in events:
        planned = getattr(e, f"{kind}_planned")
        if not planned:
            continue
        try:
            t = dt.datetime.strptime(planned, _TIME_FMT)
        except ValueError:
            continue
        diff = abs((t - target).total_seconds())
        if diff <= MATCH_TOLERANCE_SECONDS:
            candidates.append((diff, e))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])

    if len(candidates) == 1:
        return candidates[0][1]

    # multiple trains can depart/arrive within the same minute at a busy hub. A GTFS trip
    # can also occasionally not correspond to any real single live train (seen in practice:
    # a "direct" itinerary where no live ICE in that hour actually heads toward the stated
    # destination — likely a through-service chained into one trip_id in the static feed).
    # With more than one time-tolerance candidate, only trust a path-through match rather
    # than guessing — better to report no live match than confidently attach the wrong train.
    path_field = "departure_path" if kind == "departure" else "arrival_path"
    on_path = [e for _, e in candidates if _path_contains(getattr(e, path_field), through_name)]
    return on_path[0] if on_path else None


def _live_event_for(eva: str, when: dt.datetime, kind: str, through_name: str | None) -> StopEvent | None:
    try:
        plan = timetables_api.get_plan(eva, when)
    except Exception:
        return None
    match = _best_match(plan, when, kind, through_name)
    if match is None:
        return None

    try:
        changes = {e.id: e for e in timetables_api.get_full_changes(eva)}
    except Exception:
        changes = {}
    change = changes.get(match.id)
    if change:
        setattr(match, f"{kind}_changed", getattr(change, f"{kind}_changed") or getattr(match, f"{kind}_changed"))
        setattr(
            match,
            f"{kind}_platform_changed",
            getattr(change, f"{kind}_platform_changed") or getattr(match, f"{kind}_platform_changed"),
        )
        cancelled_field = f"{kind}_cancelled"
        setattr(match, cancelled_field, getattr(match, cancelled_field) or getattr(change, cancelled_field))
    return match


def _gtfs_stop_name(gtfs_con: duckdb.DuckDBPyConnection, gtfs_stop_id: int) -> str | None:
    row = gtfs_con.execute("SELECT stop_name FROM stops WHERE stop_id = ?", [gtfs_stop_id]).fetchone()
    return row[0] if row else None


def overlay_itinerary(
    gtfs_con: duckdb.DuckDBPyConnection,
    mapping: pd.DataFrame,
    itinerary: Itinerary,
    service_date: dt.date,
) -> list[LiveLeg]:
    live_legs = []
    for leg in itinerary.legs:
        board_eva = eva_for_platform(gtfs_con, mapping, leg.board_stop_id)
        alight_eva = eva_for_platform(gtfs_con, mapping, leg.alight_stop_id)
        board_name = _gtfs_stop_name(gtfs_con, leg.board_stop_id)
        alight_name = _gtfs_stop_name(gtfs_con, leg.alight_stop_id)

        board_live = (
            _live_event_for(board_eva, _seconds_to_datetime(service_date, leg.board_time), "departure", alight_name)
            if board_eva
            else None
        )
        alight_live = (
            _live_event_for(alight_eva, _seconds_to_datetime(service_date, leg.alight_time), "arrival", board_name)
            if alight_eva
            else None
        )

        live_legs.append(LiveLeg(leg, board_eva, alight_eva, board_live, alight_live))
    return live_legs
