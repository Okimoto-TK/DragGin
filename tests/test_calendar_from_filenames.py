from pathlib import Path

from src.feat.build_multiscale_tensor import build_calendar_from_daily_filenames


def test_calendar_from_filenames(tmp_path: Path) -> None:
    (tmp_path / "foo_2024-01-02.parquet").write_text("x")
    (tmp_path / "bar_20240101.parquet").write_text("x")
    (tmp_path / "bad_2024-13-01.parquet").write_text("x")
    (tmp_path / "no_date.parquet").write_text("x")

    calendar = build_calendar_from_daily_filenames(tmp_path)
    assert calendar == ["2024-01-01", "2024-01-02"]
