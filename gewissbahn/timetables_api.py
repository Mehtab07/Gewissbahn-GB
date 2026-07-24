from __future__ import annotations

import datetime as dt
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import requests

from . import config

_HEADERS = {
    "DB-Client-Id": config.DB_CLIENT_ID,
    "DB-Api-Key": config.DB_API_KEY,
    "Accept": "application/xml",
}


@dataclass
class StopEvent:
    id: str
    category: str | None = None
    train_number: str | None = None
    line: str | None = None
    arrival_planned: str | None = None
    arrival_changed: str | None = None
    arrival_platform_planned: str | None = None
    arrival_platform_changed: str | None = None
    arrival_path: list[str] = field(default_factory=list)
    arrival_cancelled: bool = False
    departure_planned: str | None = None
    departure_changed: str | None = None
    departure_platform_planned: str | None = None
    departure_platform_changed: str | None = None
    departure_path: list[str] = field(default_factory=list)
    departure_cancelled: bool = False


def normalize_eva(eva: str) -> str:
    return eva.lstrip("0") or "0"


def _parse_timetable(xml_text: str) -> list[StopEvent]:
    root = ET.fromstring(xml_text)
    events = []
    for s in root.findall("s"):
        tl = s.find("tl")
        ar = s.find("ar")
        dp = s.find("dp")
        event = StopEvent(
            id=s.get("id"),
            category=tl.get("c") if tl is not None else None,
            train_number=tl.get("n") if tl is not None else None,
            line=(ar.get("l") if ar is not None else None) or (dp.get("l") if dp is not None else None),
        )
        if ar is not None:
            event.arrival_planned = ar.get("pt")
            event.arrival_changed = ar.get("ct")
            event.arrival_platform_planned = ar.get("pp")
            event.arrival_platform_changed = ar.get("cp")
            event.arrival_path = ar.get("ppth", "").split("|") if ar.get("ppth") else []
            event.arrival_cancelled = ar.get("cs") == "c"
        if dp is not None:
            event.departure_planned = dp.get("pt")
            event.departure_changed = dp.get("ct")
            event.departure_platform_planned = dp.get("pp")
            event.departure_platform_changed = dp.get("cp")
            event.departure_path = dp.get("ppth", "").split("|") if dp.get("ppth") else []
            event.departure_cancelled = dp.get("cs") == "c"
        events.append(event)
    return events


def get_plan(eva: str, when: dt.datetime) -> list[StopEvent]:
    eva = normalize_eva(eva)
    url = f"{config.TIMETABLES_BASE_URL}/plan/{eva}/{when:%y%m%d}/{when:%H}"
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    return _parse_timetable(resp.text)


def get_full_changes(eva: str) -> list[StopEvent]:
    eva = normalize_eva(eva)
    url = f"{config.TIMETABLES_BASE_URL}/fchg/{eva}"
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    return _parse_timetable(resp.text)


def get_recent_changes(eva: str) -> list[StopEvent]:
    eva = normalize_eva(eva)
    url = f"{config.TIMETABLES_BASE_URL}/rchg/{eva}"
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    return _parse_timetable(resp.text)
