from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import duckdb
import pandas as pd

from ..gtfs import loader, station_mapping

MIN_TRANSFER_SECONDS = 120
DEFAULT_SEARCH_HORIZON_HOURS = 12


@dataclass
class Leg:
    trip_id: int
    route_id: int
    board_stop_id: int
    board_time: int
    alight_stop_id: int
    alight_time: int


@dataclass
class Itinerary:
    legs: list[Leg]

    @property
    def departure_time(self) -> int:
        return self.legs[0].board_time

    @property
    def arrival_time(self) -> int:
        return self.legs[-1].alight_time

    @property
    def n_transfers(self) -> int:
        return max(0, len(self.legs) - 1)

    def transfer_buffers(self) -> list[int]:
        """Seconds of buffer at each transfer: next leg's board_time - prev leg's alight_time."""
        return [nxt.board_time - prev.alight_time for prev, nxt in zip(self.legs, self.legs[1:])]


def _station_siblings(con: duckdb.DuckDBPyConnection) -> tuple[dict[int, list[int]], dict[int, int]]:
    df = con.execute("SELECT stop_id, coalesce(parent_station, stop_id) AS station_id FROM stops").df()
    station_platforms = df.groupby("station_id")["stop_id"].apply(list).to_dict()
    stop_to_station = dict(zip(df["stop_id"], df["station_id"]))
    return station_platforms, stop_to_station


def earliest_arrival(
    connections: pd.DataFrame,
    origin_stops: set[int],
    destination_stops: set[int],
    depart_after: int,
    station_platforms: dict[int, list[int]],
    stop_to_station: dict[int, int],
) -> Itinerary | None:
    earliest: dict[int, int] = {}
    predecessor: dict[int, tuple] = {}  # stop -> ("trip", trip_id, route_id) | ("footpath", from_stop)

    def relax(stop: int, time: int, via) -> None:
        if time < earliest.get(stop, float("inf")):
            earliest[stop] = time
            predecessor[stop] = via
            station = stop_to_station.get(stop, stop)
            for sib in station_platforms.get(station, [stop]):
                if sib == stop:
                    continue
                sib_time = time + MIN_TRANSFER_SECONDS
                if sib_time < earliest.get(sib, float("inf")):
                    earliest[sib] = sib_time
                    predecessor[sib] = ("footpath", stop)

    for s in origin_stops:
        earliest[s] = depart_after

    trip_board_stop: dict[int, int] = {}
    trip_board_time: dict[int, int] = {}
    best_dest_time = float("inf")

    conns = connections[connections["dep_time"] >= depart_after]
    for row in conns.itertuples(index=False):
        trip_id = row.trip_id
        dep_time = row.dep_time
        if dep_time > best_dest_time:
            break  # sorted by dep_time; nothing later can improve the best destination arrival

        from_stop = row.from_stop_id
        to_stop = row.to_stop_id
        arr_time = row.arr_time

        reached = trip_id in trip_board_stop
        if not reached and earliest.get(from_stop, float("inf")) <= dep_time:
            trip_board_stop[trip_id] = from_stop
            trip_board_time[trip_id] = dep_time
            reached = True

        if reached and arr_time < earliest.get(to_stop, float("inf")):
            relax(to_stop, arr_time, ("trip", trip_id, row.route_id))
            if to_stop in destination_stops and arr_time < best_dest_time:
                best_dest_time = arr_time

    if best_dest_time == float("inf"):
        return None

    dest_stop = next(s for s in destination_stops if earliest.get(s) == best_dest_time)

    legs: list[Leg] = []
    cursor = dest_stop
    while cursor not in origin_stops:
        via = predecessor.get(cursor)
        if via is None:
            break
        if via[0] == "trip":
            _, trip_id, route_id = via
            board_stop = trip_board_stop[trip_id]
            board_time = trip_board_time[trip_id]
            legs.append(Leg(trip_id, route_id, board_stop, board_time, cursor, earliest[cursor]))
            cursor = board_stop
        else:
            _, from_stop = via
            cursor = from_stop
    legs.reverse()

    return Itinerary(legs=legs) if legs else None


def find_itineraries(
    origin_eva: str,
    destination_eva: str,
    depart_after: dt.datetime,
    service_date: dt.date,
    count: int = 3,
    search_horizon_hours: int = DEFAULT_SEARCH_HORIZON_HOURS,
    gtfs_con: duckdb.DuckDBPyConnection | None = None,
) -> list[Itinerary]:
    con = gtfs_con or loader.connect()
    mapping = station_mapping.get_mapping(con)

    origin_row = mapping[(mapping["eva"] == origin_eva) & mapping["gtfs_stop_id"].notna()]
    dest_row = mapping[(mapping["eva"] == destination_eva) & mapping["gtfs_stop_id"].notna()]
    if origin_row.empty or dest_row.empty:
        return []

    origin_station = str(int(origin_row.iloc[0]["gtfs_stop_id"]))
    dest_station = str(int(dest_row.iloc[0]["gtfs_stop_id"]))
    origin_stops = {int(x) for x in station_mapping.platform_stop_ids(con, origin_station)}
    dest_stops = {int(x) for x in station_mapping.platform_stop_ids(con, dest_station)}

    station_platforms, stop_to_station = _station_siblings(con)

    depart_secs = depart_after.hour * 3600 + depart_after.minute * 60 + depart_after.second
    horizon_secs = depart_secs + search_horizon_hours * 3600

    connections = loader.connections_for_date(con, service_date)
    connections = connections[connections["dep_time"] <= horizon_secs]

    itineraries: list[Itinerary] = []
    cutoff = depart_secs
    for _ in range(count):
        it = earliest_arrival(connections, origin_stops, dest_stops, cutoff, station_platforms, stop_to_station)
        if it is None:
            break
        itineraries.append(it)
        cutoff = it.departure_time + 60

    return itineraries
