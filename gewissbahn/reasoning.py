from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI

from . import config

_SYSTEM_PROMPT = (
    "You are explaining train journey options to a traveller. For each option you are "
    "given its schedule and a historical reliability confidence score already computed by "
    "a statistical model. Write a short, plain-language explanation per option (1-2 "
    "sentences) that uses ONLY the numbers given to you -- never invent a statistic. "
    "The confidence score reflects historical transfer reliability ONLY -- it says nothing "
    "about whether the itinerary corresponds to a real, currently-operating train. An "
    "option with zero transfers and NO live confirmation on any leg is NOT automatically "
    "the best choice: treat 'no live confirmation at all' as a real red flag that the "
    "route may not exist as scheduled, not a neutral gap in the data, and weigh it "
    "explicitly against options that do have live confirmation, even if those have a "
    "lower raw confidence score. End with a one-line overall recommendation naming the "
    "best option and why."
)


@dataclass
class ItinerarySummary:
    label: str
    departure: str
    arrival: str
    duration_min: int
    n_transfers: int
    confidence: float
    transfer_details: list[str] = field(default_factory=list)
    live_note: str | None = None

    def as_prompt_block(self) -> str:
        lines = [
            f"{self.label}: depart {self.departure}, arrive {self.arrival} "
            f"({self.duration_min} min), {self.n_transfers} transfer(s), "
            f"confidence {self.confidence:.0%}"
        ]
        lines += [f"  - {d}" for d in self.transfer_details]
        if self.live_note:
            lines.append(f"  - live data: {self.live_note}")
        return "\n".join(lines)


def _client() -> OpenAI:
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env")
    return OpenAI(api_key=config.OPENROUTER_API_KEY, base_url=config.OPENROUTER_BASE_URL)


def explain(summaries: list[ItinerarySummary], origin: str, destination: str) -> str:
    if not summaries:
        return "No itineraries found."

    prompt = f"Journey: {origin} to {destination}\n\n" + "\n\n".join(s.as_prompt_block() for s in summaries)

    client = _client()
    response = client.chat.completions.create(
        model=config.REASONING_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content
