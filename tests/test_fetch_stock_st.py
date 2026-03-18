from pathlib import Path

import pandas as pd

from tools import fetch_stock_st


class FakePro:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def trade_cal(self, **kwargs):
        self.calls.append({"api": "trade_cal", **kwargs})
        return pd.DataFrame({"cal_date": ["20250813", "20250814"], "is_open": [1, 1]})

    def stock_st(self, **kwargs):
        self.calls.append({"api": "stock_st", **kwargs})
        trade_date = str(kwargs["trade_date"])
        offset = int(kwargs.get("offset", 0))
        if trade_date == "20250813" and offset == 0:
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "name": ["ST平安"],
                    "start_date": ["20250813"],
                    "end_date": [None],
                    "change_reason": ["risk_warning"],
                }
            )
        if trade_date == "20250814" and offset == 0:
            return pd.DataFrame(
                {
                    "ts_code": ["000002.SZ", "000003.SZ"],
                    "name": ["*ST万科", "ST样本"],
                    "start_date": ["20250801", "20250810"],
                    "end_date": [None, None],
                    "change_reason": ["risk_warning", "risk_warning"],
                }
            )
        return pd.DataFrame()


def test_fetch_stock_st_by_date_normalizes_and_deduplicates() -> None:
    class SingleDayPro:
        def __init__(self) -> None:
            self.calls = 0

        def stock_st(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return pd.DataFrame(
                    {
                        "ts_code": ["000001.SZ", "000001.SZ"],
                        "name": ["ST平安", "ST平安"],
                        "start_date": ["20250813", "20250813"],
                        "end_date": [None, None],
                        "change_reason": ["risk_warning", "risk_warning"],
                    }
                )
            return pd.DataFrame()

    df = fetch_stock_st._fetch_stock_st_by_date(SingleDayPro(), trade_date="20250813")

    assert df.columns.tolist() == ["trade_date", "ts_code", "name", "start_date", "end_date", "change_reason"]
    assert df.shape == (1, 6)
    assert df.loc[0, "trade_date"] == "20250813"
    assert df.loc[0, "ts_code"] == "000001.SZ"


def test_fetch_stock_st_range_writes_one_parquet_per_date(tmp_path: Path, monkeypatch) -> None:
    fake_pro = FakePro()

    monkeypatch.setattr(fetch_stock_st, "_create_pro_client", lambda token: fake_pro)
    monkeypatch.setattr(fetch_stock_st, "_worker_main", lambda token, out_dir, trade_date, sleep_seconds, max_retries, error_queue: pd.DataFrame({
        "trade_date": [trade_date],
        "ts_code": [f"{trade_date}.SZ"],
        "name": ["ST样本"],
        "start_date": [trade_date],
        "end_date": [None],
        "change_reason": ["risk_warning"],
    }).to_parquet(Path(out_dir) / f"{trade_date}_stock_st.parquet", index=False) or error_queue.put((trade_date, None)))

    class InlineProcess:
        def __init__(self, target, args):
            self.target = target
            self.args = args
            self.exitcode = None

        def start(self):
            self.target(*self.args)
            self.exitcode = 0

        def join(self):
            return None

        def terminate(self):
            self.exitcode = -15

    class InlineContext:
        def Queue(self):
            from queue import Queue
            return Queue()

        def Process(self, target, args):
            return InlineProcess(target, args)

    monkeypatch.setattr(fetch_stock_st.mp, "get_all_start_methods", lambda: ["fork"])
    monkeypatch.setattr(fetch_stock_st.mp, "get_context", lambda method: InlineContext())

    fetch_stock_st.fetch_stock_st_range(
        start_date="20250813",
        end_date="20250814",
        out_dir=tmp_path,
        token="demo-token",
        max_workers=2,
    )

    first = pd.read_parquet(tmp_path / "20250813_stock_st.parquet")
    second = pd.read_parquet(tmp_path / "20250814_stock_st.parquet")

    assert first["trade_date"].tolist() == ["20250813"]
    assert second["trade_date"].tolist() == ["20250814"]
    assert fake_pro.calls[0]["api"] == "trade_cal"
