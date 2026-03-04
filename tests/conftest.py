import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import date, timedelta


class FakeProvider:
    def __init__(self, data):
        self.data = data

    def get_bars(self, code, interval, end_date, total_bars):
        bars = [b for b in self.data[(code, interval)] if b["date"] <= end_date]
        return bars[-total_bars:]


def make_calendar(days=80, start=date(2025, 1, 1)):
    return [start + timedelta(days=i) for i in range(days)]


def make_bars(calendar, bars_per_day, base=100.0):
    bars = []
    px = base
    for d in calendar:
        for _ in range(bars_per_day):
            px += 0.1
            bars.append(
                {
                    "date": d,
                    "open": px,
                    "high": px + 0.2,
                    "low": px - 0.2,
                    "close": px + 0.05,
                    "volume": 1000.0,
                    "vwap": px + 0.03,
                }
            )
    return bars


def make_provider(calendar):
    code = "AAA"
    data = {
        (code, "1d"): make_bars(calendar, 1),
        (code, "30m"): make_bars(calendar, 8),
        (code, "5m"): make_bars(calendar, 48),
    }
    return code, FakeProvider(data)
