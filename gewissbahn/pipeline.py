from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

import duckdb
import pandas as pd

from . import historical, reasoning
from .gtfs import loader, station_mapping
from .routing import csa, live_overlay

_UMLAUT_FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
_NON_WORD_RE = re.compile(r"[^a-z0-9]+")


def _fmt_time(secs: int) -> str:
    secs = secs % 86400
    return f"{secs // 3600:02d}:{(secs % 3600) // 60:02d}"


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


def _station_name(mapping: pd.DataFrame, eva: str | None) -> str:
    if eva is None:
        return "unknown station"
    hit = mapping[mapping["eva"] == eva]
    return hit.iloc[0]["station_name"] if not hit.empty else eva


def _train_label(event) -> str:
    if event is None:
        return "train (no live match)"
    parts = [p for p in (event.category, event.train_number) if p]
    label = " ".join(parts) if parts else (event.line or "train")
    if event.line and event.line not in label:
        label += f" [{event.line}]"
    return label


def _leg_status(live_leg) -> str:
    # DB's API only sends a "changed" time when a leg actually deviates from schedule, so
    # delay_min is None both when we have no live match AND when the leg is exactly on
    # time -- check whether we matched a live event at all to tell those apart.
    bits = []
    if live_leg.board_live is not None:
        bits.append(f"dep +{live_leg.board_delay_min}min" if live_leg.board_delay_min else "dep on time")
    if live_leg.alight_live is not None:
        bits.append(f"arr +{live_leg.alight_delay_min}min" if live_leg.alight_delay_min else "arr on time")
    if live_leg.is_cancelled:
        bits.append("CANCELLED")
    return ", ".join(bits) if bits else "no live match"


def _build_summary(index: int, itinerary, score, live_legs, mapping: pd.DataFrame) -> reasoning.ItinerarySummary:
    has_live = any(ll.board_live or ll.alight_live for ll in live_legs)
    delays = [ll.board_delay_min for ll in live_legs if ll.board_delay_min is not None]
    if not has_live:
        live_note = "no live confirmation found for this itinerary"
    elif delays:
        live_note = f"live data shows current delays of {delays} min on matched legs"
    else:
        live_note = "live confirmed, on schedule"

    leg_details = []
    for leg, live_leg in zip(itinerary.legs, live_legs):
        board_name = _station_name(mapping, live_leg.board_eva)
        alight_name = _station_name(mapping, live_leg.alight_eva)
        train = _train_label(live_leg.board_live or live_leg.alight_live)
        leg_details.append(
            f"{train}: {board_name} {_fmt_time(leg.board_time)} -> {alight_name} {_fmt_time(leg.alight_time)} "
            f"({_leg_status(live_leg)})"
        )

    details = [
        f"transfer at {_station_name(mapping, t.eva)}: {t.success_probability:.0%} historical success (n={t.sample_size})"
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
        leg_details=leg_details,
        transfer_details=details,
        live_note=live_note,
    )


@dataclass
class PlanResult:
    summaries: list[reasoning.ItinerarySummary]
    explanation: str | None = None
    error: str | None = None


def plan_journey(
    origin: str,
    destination: str,
    when: dt.datetime,
    count: int = 3,
    gtfs_con: duckdb.DuckDBPyConnection | None = None,
    mapping: pd.DataFrame | None = None,
) -> PlanResult:
    """Runs the full pipeline (routing -> live overlay -> historical scoring -> LLM
    explanation) and returns structured results, no printing -- callers (CLI, UI) decide
    how to present it. `gtfs_con`/`mapping` can be passed in to reuse an already-loaded
    connection (e.g. cached across requests in a UI) instead of reconnecting each call."""
    gtfs_con = gtfs_con or loader.connect()
    mapping = mapping if mapping is not None else station_mapping.get_mapping(gtfs_con)
    hist_con = duckdb.connect()
    service_date = when.date()

    try:
        origin_eva = resolve_station(mapping, origin)
        destination_eva = resolve_station(mapping, destination)
    except ValueError as e:
        return PlanResult(summaries=[], error=str(e))

    itineraries = csa.find_itineraries(
        origin_eva=origin_eva,
        destination_eva=destination_eva,
        depart_after=when,
        service_date=service_date,
        count=count,
        gtfs_con=gtfs_con,
    )
    if not itineraries:
        return PlanResult(
            summaries=[],
            error=f"No itineraries found from {origin} to {destination} after {when:%Y-%m-%d %H:%M}.",
        )

    summaries = []
    for index, itinerary in enumerate(itineraries):
        score = historical.score_itinerary(hist_con, gtfs_con, mapping, itinerary, service_date)
        live_legs = live_overlay.overlay_itinerary(gtfs_con, mapping, itinerary, service_date)
        summaries.append(_build_summary(index, itinerary, score, live_legs, mapping))

    explanation = reasoning.explain(summaries, origin=origin, destination=destination)
    return PlanResult(summaries=summaries, explanation=explanation)
