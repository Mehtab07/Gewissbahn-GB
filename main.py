from __future__ import annotations

import argparse
import datetime as dt

from gewissbahn.pipeline import plan_journey


def main():
    parser = argparse.ArgumentParser(description="Gewissbahn-GB: reliability-aware German train routing")
    parser.add_argument("origin", help="Origin station name, e.g. 'Koeln Hbf'")
    parser.add_argument("destination", help="Destination station name, e.g. 'Berlin Hbf'")
    parser.add_argument("--when", help="Departure date/time, ISO format e.g. '2026-07-24 10:00' (default: now)")
    parser.add_argument("--count", type=int, default=3, help="Number of itinerary options to find (default: 3)")
    args = parser.parse_args()

    when = dt.datetime.fromisoformat(args.when) if args.when else dt.datetime.now()

    result = plan_journey(args.origin, args.destination, when, count=args.count)

    if result.error:
        print(f"Error: {result.error}")
        return

    print(f"\nFound {len(result.summaries)} option(s) from {args.origin} to {args.destination}:\n")
    for s in result.summaries:
        print(s.as_prompt_block())
        print()

    print("--- Recommendation ---")
    print(result.explanation)


if __name__ == "__main__":
    main()
