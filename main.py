from __future__ import annotations

import argparse
import datetime as dt
import re

import duckdb
import pandas as pd

from gewissbahn import historical, reasoning
from gewissbahn.gtfs import loader, station_mapping
from gewissbahn.routing import csa, live_overlay


def _fmt_time(secs: int) -> str:
    secs = secs % 86400
    return f"{secs // 3600:02d}:{(secs % 3600) // 60:02d}"


_UMLAUT_FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
_NON_WORD_RE = re.compile(r"[^a-z0-9]+")


def _fold(s: str) -> str:
    """Lowercase + fold umlauts to their ASCII spelling (oe/ue/ae/ss), so a plain-ASCII
    query like "Koeln" matches data stored as "Köln"."""
    return s.strip().lower().translate(_UMLAUT_FOLD)


def _tokens(s: str) -> set[str]:
    return set(_NON_WORD_RE.sub(" ", _fold(s)).split())


def resolve_station(mapping: pd.DataFrame, query: str) -> str:
    """Find an eva by station name. Matches on the set of words in the query being present
    in the candidate name, not a literal substring -- real station names often wedge extra
    text between words (e.g. "Frankfurt(Main)Hbf"), which breaks naive substring search for
    the natural query "Frankfurt Hbf" and can silently resolve to the wrong station (this
    happened in testing: it matched "Frankfurt Hbf (tief)", an S-Bahn tunnel stop, instead
    of the actual long-distance hub)."""
    query_tokens = _tokens(query)
    if not query_tokens:
        raise ValueError(f'No station found matching "{query}"')

    name_tokens = mapping["station_name"].map(_tokens)
    exact = mapping[name_tokens == query_tokens]
    if not exact.empty:
        return exact.iloc[0]["eva"]

    hits = mapping[name_tokens.map(lambda t: query_tokens.issubset(t))]
    if not hits.empty:
        # prefer the shortest matching name -- the main station over a qualified variant
        # like "(S-Bahn)"/"(tief)"
        hits = hits.assign(_len=hits["station_name"].str.len())
        return hits.sort_values("_len").iloc[0]["eva"]

    raise ValueError(f'No station found matching "{query}"')


def _build_summary(index: int, itinerary, score, live_legs) -> reasoning.ItinerarySummary:
    has_live = any(ll.board_live or ll.alight_live for ll in live_legs)
    delays = [ll.board_delay_min for ll in live_legs if ll.board_delay_min is not None]
    if not has_live:
        live_note = "no live confirmation found for this itinerary"
    elif delays:
        live_note = f"live data shows current delays of {delays} min on matched legs"
    else:
        live_note = "live confirmed, on schedule"

    details = [
        f"transfer at {t.eva}: {t.success_probability:.0%} historical success (n={t.sample_size})"
        for t in score.transfers
        if t.eva
    ]

    return reasoning.ItinerarySummary(
        label=f"Option {index + 1}",
        departure=_fmt_time(itinerary.departure_time),
        arrival=_fmt_time(itinerary.arrival_time),
        duration_min=(itinerary.arrival_time - itinerary.departure_time) // 60,
        n_transfers=itinerary.n_transfers,
        confidence=score.confidence,
        transfer_details=details,
        live_note=live_note,
    )


def plan_journey(origin: str, destination: str, when: dt.datetime, count: int = 3) -> str:
    gtfs_con = loader.connect()
    mapping = station_mapping.get_mapping(gtfs_con)
    hist_con = duckdb.connect()
    service_date = when.date()

    origin_eva = resolve_station(mapping, origin)
    destination_eva = resolve_station(mapping, destination)

    itineraries = csa.find_itineraries(
        origin_eva=origin_eva,
        destination_eva=destination_eva,
        depart_after=when,
        service_date=service_date,
        count=count,
        gtfs_con=gtfs_con,
    )
    if not itineraries:
        return f"No itineraries found from {origin} to {destination} after {when:%Y-%m-%d %H:%M}."

    summaries = []
    for index, itinerary in enumerate(itineraries):
        score = historical.score_itinerary(hist_con, gtfs_con, mapping, itinerary, service_date)
        live_legs = live_overlay.overlay_itinerary(gtfs_con, mapping, itinerary, service_date)
        summaries.append(_build_summary(index, itinerary, score, live_legs))

    print(f"\nFound {len(summaries)} option(s) from {origin} to {destination}:\n")
    for s in summaries:
        print(s.as_prompt_block())
        print()

    return reasoning.explain(summaries, origin=origin, destination=destination)


def main():
    parser = argparse.ArgumentParser(description="Gewissbahn-GB: reliability-aware German train routing")
    parser.add_argument("origin", help="Origin station name, e.g. 'Koeln Hbf'")
    parser.add_argument("destination", help="Destination station name, e.g. 'Berlin Hbf'")
    parser.add_argument("--when", help="Departure date/time, ISO format e.g. '2026-07-24 10:00' (default: now)")
    parser.add_argument("--count", type=int, default=3, help="Number of itinerary options to find (default: 3)")
    args = parser.parse_args()

    when = dt.datetime.fromisoformat(args.when) if args.when else dt.datetime.now()

    try:
        explanation = plan_journey(args.origin, args.destination, when, count=args.count)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print("--- Recommendation ---")
    print(explanation)


if __name__ == "__main__":
    main()
