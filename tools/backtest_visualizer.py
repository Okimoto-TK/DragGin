from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any



PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _load_backtest_rows(backtest_dir: Path) -> list[dict[str, Any]]:
    if not backtest_dir.exists():
        raise FileNotFoundError(f"backtest dir not found: {backtest_dir}")

    rows: list[dict[str, Any]] = []
    for path in sorted(backtest_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        date = str(payload.get("date") or path.stem)
        final_total_asset = float(payload.get("final_total_asset", 0.0))
        initial_total_asset = float(payload.get("initial_total_asset", final_total_asset))
        daily_return = 0.0
        if initial_total_asset > 0:
            daily_return = final_total_asset / initial_total_asset - 1.0
        rows.append(
            {
                "date": date,
                "asof_date": str(payload.get("asof_date", "")),
                "initial_total_asset": initial_total_asset,
                "final_total_asset": final_total_asset,
                "daily_return": daily_return,
                "payload": payload,
            }
        )

    if not rows:
        raise ValueError(f"no backtest json files found in: {backtest_dir}")

    base_asset = rows[0]["initial_total_asset"]
    if base_asset <= 0:
        base_asset = rows[0]["final_total_asset"]
    if base_asset <= 0:
        raise ValueError("cannot compute cumulative return because initial asset is non-positive")

    for row in rows:
        row["cum_return"] = row["final_total_asset"] / base_asset - 1.0
    return rows


def _build_html(rows: list[dict[str, Any]], title: str) -> str:
    serializable_rows = []
    for row in rows:
        serializable_rows.append(
            {
                "date": row["date"],
                "asof_date": row["asof_date"],
                "initial_total_asset": row["initial_total_asset"],
                "final_total_asset": row["final_total_asset"],
                "daily_return": row["daily_return"],
                "cum_return": row["cum_return"],
                "payload": row["payload"],
            }
        )

    rows_json = json.dumps(serializable_rows, ensure_ascii=False)
    title_text = html.escape(title)
    return f"""<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title_text}</title>
  <script src=\"{PLOTLY_CDN}\"></script>
  <style>
    :root {{
      color-scheme: light;
      --border: #d9d9d9;
      --bg-soft: #f7f7f9;
      --text-main: #1f2328;
      --text-muted: #667085;
      --positive: #0f9d58;
      --negative: #d93025;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text-main); background: #fff; }}
    .page {{ display: flex; min-height: 100vh; }}
    .main {{ flex: 1 1 auto; padding: 16px; min-width: 0; }}
    .sidebar {{ width: 420px; max-width: 42vw; border-left: 1px solid var(--border); background: var(--bg-soft); padding: 16px; overflow: auto; }}
    .headline {{ margin: 0 0 8px; font-size: 22px; }}
    .subhead {{ margin: 0 0 16px; color: var(--text-muted); font-size: 14px; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .card {{ background: #fff; border: 1px solid var(--border); border-radius: 10px; padding: 12px; }}
    .card .label {{ font-size: 12px; color: var(--text-muted); margin-bottom: 6px; }}
    .card .value {{ font-size: 18px; font-weight: 600; }}
    #chart {{ width: 100%; height: calc(100vh - 210px); min-height: 520px; border: 1px solid var(--border); border-radius: 10px; }}
    .sidebar h2 {{ margin: 0 0 12px; font-size: 18px; }}
    .meta-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }}
    .meta {{ background: #fff; border: 1px solid var(--border); border-radius: 10px; padding: 10px; }}
    .meta .label {{ font-size: 12px; color: var(--text-muted); margin-bottom: 4px; }}
    .meta .value {{ font-size: 14px; font-weight: 600; word-break: break-word; }}
    .json-box {{ white-space: pre-wrap; word-break: break-word; background: #111827; color: #e5e7eb; border-radius: 10px; padding: 14px; font-size: 12px; line-height: 1.55; overflow-x: auto; }}
    .hint {{ font-size: 12px; color: var(--text-muted); margin-bottom: 12px; }}
    .positive {{ color: var(--positive); }}
    .negative {{ color: var(--negative); }}
    @media (max-width: 1100px) {{
      .page {{ flex-direction: column; }}
      .sidebar {{ width: 100%; max-width: none; border-left: none; border-top: 1px solid var(--border); }}
      #chart {{ height: 70vh; min-height: 420px; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <div class=\"page\">
    <div class=\"main\">
      <h1 class=\"headline\">{title_text}</h1>
      <p class=\"subhead\">支持 Plotly 原生缩放、框选、平移与重置。点击曲线上的任意日期，可在右侧查看当日完整 JSON 明细。</p>
      <div class=\"stats\">
        <div class=\"card\"><div class=\"label\">交易日数量</div><div class=\"value\" id=\"stat-days\"></div></div>
        <div class=\"card\"><div class=\"label\">期初总资产</div><div class=\"value\" id=\"stat-start\"></div></div>
        <div class=\"card\"><div class=\"label\">期末总资产</div><div class=\"value\" id=\"stat-end\"></div></div>
        <div class=\"card\"><div class=\"label\">累计收益率</div><div class=\"value\" id=\"stat-return\"></div></div>
      </div>
      <div id=\"chart\"></div>
    </div>
    <aside class=\"sidebar\">
      <h2>当日明细</h2>
      <div class=\"hint\">默认显示最后一个交易日；点击图上数据点可切换。</div>
      <div class=\"meta-grid\">
        <div class=\"meta\"><div class=\"label\">交易日</div><div class=\"value\" id=\"detail-date\"></div></div>
        <div class=\"meta\"><div class=\"label\">asof_date</div><div class=\"value\" id=\"detail-asof\"></div></div>
        <div class=\"meta\"><div class=\"label\">日收益率</div><div class=\"value\" id=\"detail-daily-return\"></div></div>
        <div class=\"meta\"><div class=\"label\">累计收益率</div><div class=\"value\" id=\"detail-cum-return\"></div></div>
      </div>
      <div class=\"json-box\" id=\"detail-json\"></div>
    </aside>
  </div>

  <script>
    const rows = {rows_json};

    function formatMoney(v) {{
      return new Intl.NumberFormat('zh-CN', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }}).format(v);
    }}

    function formatPct(v) {{
      return `${{(v * 100).toFixed(2)}}%`;
    }}

    function setValue(id, text, cls='') {{
      const el = document.getElementById(id);
      el.textContent = text;
      el.className = `value ${{cls}}`.trim();
    }}

    function updateSummary() {{
      const first = rows[0];
      const last = rows[rows.length - 1];
      document.getElementById('stat-days').textContent = String(rows.length);
      document.getElementById('stat-start').textContent = formatMoney(first.initial_total_asset);
      document.getElementById('stat-end').textContent = formatMoney(last.final_total_asset);
      const cls = last.cum_return >= 0 ? 'positive' : 'negative';
      const ret = document.getElementById('stat-return');
      ret.textContent = formatPct(last.cum_return);
      ret.className = `value ${{cls}}`;
    }}

    function updateDetail(idx) {{
      const row = rows[idx];
      document.getElementById('detail-date').textContent = row.date;
      document.getElementById('detail-asof').textContent = row.asof_date || '-';
      setValue('detail-daily-return', formatPct(row.daily_return), row.daily_return >= 0 ? 'positive' : 'negative');
      setValue('detail-cum-return', formatPct(row.cum_return), row.cum_return >= 0 ? 'positive' : 'negative');
      document.getElementById('detail-json').textContent = JSON.stringify(row.payload, null, 2);
    }}

    updateSummary();

    const trace = {{
      x: rows.map(r => r.date),
      y: rows.map(r => r.cum_return * 100),
      customdata: rows.map((r, idx) => [idx, r.daily_return * 100, r.final_total_asset]),
      type: 'scatter',
      mode: 'lines+markers',
      line: {{ color: '#2563eb', width: 2 }},
      marker: {{ color: '#2563eb', size: 7 }},
      hovertemplate: [
        '日期: %{{x}}',
        '累计收益率: %{{y:.2f}}%',
        '日收益率: %{{customdata[1]:.2f}}%',
        '期末总资产: %{{customdata[2]:,.2f}}',
        '<extra></extra>'
      ].join('<br>')
    }};

    const layout = {{
      margin: {{ l: 60, r: 20, t: 30, b: 60 }},
      hovermode: 'closest',
      dragmode: 'zoom',
      xaxis: {{ title: '交易日', rangeslider: {{ visible: true }} }},
      yaxis: {{ title: '累计收益率 (%)', ticksuffix: '%' }},
      template: 'plotly_white'
    }};

    Plotly.newPlot('chart', [trace], layout, {{ responsive: true, displaylogo: false }}).then((plot) => {{
      plot.on('plotly_click', (event) => {{
        if (!event.points || !event.points.length) return;
        const idx = event.points[0].customdata[0];
        updateDetail(idx);
      }});
    }});

    updateDetail(rows.length - 1);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an interactive HTML viewer for backtest daily json outputs")
    parser.add_argument("--backtest-dir", required=True, help="directory containing daily backtest json files")
    parser.add_argument("--out-file", default="", help="output html file path; defaults to <backtest-dir>/backtest_report.html")
    parser.add_argument("--title", default="Backtest Daily Return Viewer")
    args = parser.parse_args()

    backtest_dir = Path(args.backtest_dir)
    out_file = Path(args.out_file) if args.out_file else backtest_dir / "backtest_report.html"
    rows = _load_backtest_rows(backtest_dir)
    html_text = _build_html(rows, args.title)
    out_file.write_text(html_text, encoding="utf-8")
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
