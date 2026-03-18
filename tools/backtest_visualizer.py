from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
SHORT_WINDOW_THRESHOLD = 45


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
        row["start_cum_return"] = row["initial_total_asset"] / base_asset - 1.0
        row["cum_return"] = row["final_total_asset"] / base_asset - 1.0
    return rows


def _build_html(rows: list[dict[str, Any]], title: str) -> str:
    serializable_rows = [
        {
            "date": row["date"],
            "asof_date": row["asof_date"],
            "initial_total_asset": row["initial_total_asset"],
            "final_total_asset": row["final_total_asset"],
            "daily_return": row["daily_return"],
            "start_cum_return": row["start_cum_return"],
            "cum_return": row["cum_return"],
            "payload": row["payload"],
        }
        for row in rows
    ]

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
      --border: #e5e7eb;
      --border-strong: #d0d5dd;
      --bg-page: #f8fafc;
      --bg-card: rgba(255,255,255,0.88);
      --text-main: #0f172a;
      --text-muted: #667085;
      --text-soft: #475467;
      --positive: #d92d20;
      --negative: #1570ef;
      --shadow: 0 14px 28px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text-main); background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%); }}
    .page {{ display: flex; min-height: 100vh; }}
    .main {{ flex: 1 1 auto; min-width: 0; padding: 20px; }}
    .sidebar {{ width: 460px; max-width: 44vw; border-left: 1px solid rgba(208, 213, 221, 0.75); background: rgba(248, 250, 252, 0.92); backdrop-filter: blur(14px); padding: 20px; overflow: auto; }}
    .panel {{ background: var(--bg-card); border: 1px solid rgba(229, 231, 235, 0.95); border-radius: 18px; box-shadow: var(--shadow); }}
    .hero {{ padding: 18px 20px; margin-bottom: 16px; }}
    .headline {{ margin: 0 0 8px; font-size: 24px; font-weight: 700; }}
    .subhead {{ margin: 0; color: var(--text-muted); font-size: 14px; line-height: 1.6; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .stat-card {{ padding: 14px 16px; }}
    .label {{ font-size: 12px; color: var(--text-muted); margin-bottom: 6px; }}
    .value {{ font-size: 18px; font-weight: 700; color: var(--text-main); }}
    .positive {{ color: var(--positive) !important; }}
    .negative {{ color: var(--negative) !important; }}
    .chart-panel {{ padding: 12px; }}
    #chart {{ width: 100%; height: calc(100vh - 220px); min-height: 560px; border-radius: 14px; }}
    .sidebar-title {{ margin: 0 0 6px; font-size: 20px; font-weight: 700; }}
    .sidebar-subtitle {{ margin: 0 0 16px; color: var(--text-muted); font-size: 13px; line-height: 1.6; }}
    .detail-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-bottom: 14px; }}
    .detail-card {{ padding: 12px 14px; }}
    .section {{ margin-top: 16px; }}
    .section-title {{ margin: 0 0 10px; font-size: 14px; font-weight: 700; color: var(--text-soft); }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .metric-card {{ padding: 12px 14px; }}
    .list {{ display: grid; gap: 10px; }}
    .trade-card {{ padding: 12px 14px; }}
    .trade-card-head {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 10px; }}
    .trade-code {{ font-size: 14px; font-weight: 700; }}
    .pill {{ border-radius: 999px; padding: 4px 9px; font-size: 11px; font-weight: 700; background: #eef4ff; color: #175cd3; }}
    .kv-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 12px; }}
    .kv {{ min-width: 0; }}
    .kv .label {{ margin-bottom: 2px; }}
    .kv .text {{ font-size: 13px; color: var(--text-main); font-weight: 600; word-break: break-word; }}
    .empty {{ padding: 14px; color: var(--text-muted); font-size: 13px; text-align: center; border: 1px dashed var(--border-strong); border-radius: 14px; background: rgba(255,255,255,0.7); }}
    @media (max-width: 1200px) {{
      .page {{ flex-direction: column; }}
      .sidebar {{ width: 100%; max-width: none; border-left: none; border-top: 1px solid rgba(208, 213, 221, 0.75); }}
      #chart {{ height: 72vh; min-height: 460px; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <div class=\"page\">
    <div class=\"main\">
      <section class=\"panel hero\">
        <h1 class=\"headline\">{title_text}</h1>
        <p class=\"subhead\">默认显示累计收益视图。缩放后会自动重算纵轴范围；区间较大时显示平滑曲线，区间缩短后自动切换成无影线 K 线（上涨红色、下跌蓝色），并在悬浮时展示当日细节。</p>
      </section>
      <section class=\"stats\">
        <div class=\"panel stat-card\"><div class=\"label\">交易日数量</div><div class=\"value\" id=\"stat-days\"></div></div>
        <div class=\"panel stat-card\"><div class=\"label\">期初总资产</div><div class=\"value\" id=\"stat-start\"></div></div>
        <div class=\"panel stat-card\"><div class=\"label\">期末总资产</div><div class=\"value\" id=\"stat-end\"></div></div>
        <div class=\"panel stat-card\"><div class=\"label\">累计收益率</div><div class=\"value\" id=\"stat-return\"></div></div>
      </section>
      <section class=\"panel chart-panel\">
        <div id=\"chart\"></div>
      </section>
    </div>
    <aside class="sidebar">
      <h2 class="sidebar-title">当日明细</h2>
      <p class="sidebar-subtitle">默认展示最后一个交易日。点击图上的任意点或柱体，可在这里查看结构化的交易摘要、持仓变化与买卖记录。</p>
      <div class="sidebar-stage">
        <div class="sidebar-track" id="sidebar-track">
          <section class="sidebar-pane">
            <div class="detail-grid">
              <div class="panel detail-card"><div class="label">交易日</div><div class="value" id="detail-date"></div></div>
              <div class="panel detail-card"><div class="label">asof_date</div><div class="value" id="detail-asof"></div></div>
              <div class="panel detail-card"><div class="label">日收益率</div><div class="value" id="detail-daily-return"></div></div>
              <div class="panel detail-card"><div class="label">累计收益率</div><div class="value" id="detail-cum-return"></div></div>
            </div>
            <div id="detail-summary"></div>
            <div class="section-button-grid" id="section-buttons"></div>
          </section>
          <section class="sidebar-pane">
            <div class="detail-shell">
              <div class="detail-header">
                <button class="back-button" id="detail-back" type="button">← 返回</button>
                <div class="detail-panel-title" id="detail-panel-title">详情</div>
              </div>
              <div id="detail-panel-content"></div>
            </div>
          </section>
        </div>
      </div>
    </aside>
  </div>
  <script>
    const rows = {rows_json};
    const SHORT_WINDOW_THRESHOLD = {SHORT_WINDOW_THRESHOLD};
    let syncLock = false;

    function formatMoney(v) {{
      const num = Number(v || 0);
      return new Intl.NumberFormat('zh-CN', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }}).format(num);
    }}

    function formatPct(v) {{
      return `${{(Number(v || 0) * 100).toFixed(2)}}%`;
    }}

    function escapeHtml(value) {{
      return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }}

    function setValue(id, text, cls='') {{
      const el = document.getElementById(id);
      el.textContent = text;
      el.className = `value ${{cls}}`.trim();
    }}

    function sectionHtml(title, innerHtml) {{
      return `<section class="section"><h3 class="section-title">${{title}}</h3>${{innerHtml}}</section>`;
    }}

    function metricCard(label, value, cls='') {{
      return `<div class="panel metric-card"><div class="label">${{escapeHtml(label)}}</div><div class="value ${{cls}}">${{escapeHtml(value)}}</div></div>`;
    }}

    function emptyState(text) {{
      return `<div class="empty">${{escapeHtml(text)}}</div>`;
    }}

    function recordCard(record, kindLabel) {{
      const items = Object.entries(record || {{}})
        .filter(([key]) => key !== 'code')
        .map(([key, value]) => `
          <div class="kv">
            <div class="label">${{escapeHtml(key)}}</div>
            <div class="text">${{typeof value === 'number' ? escapeHtml(formatMoney(value)) : escapeHtml(String(value))}}</div>
          </div>
        `)
        .join('');
      return `
        <article class="panel trade-card">
          <div class="trade-card-head">
            <div class="trade-code">${{escapeHtml(record.code || '-')}}</div>
            <span class="pill">${{escapeHtml(kindLabel)}}</span>
          </div>
          <div class="kv-grid">${{items}}</div>
        </article>
      `;
    }}

    function recordsSection(title, records, emptyText, kindLabel) {{
      if (!records || !records.length) return sectionHtml(title, emptyState(emptyText));
      return sectionHtml(title, `<div class="list">${{records.map(row => recordCard(row, kindLabel)).join('')}}</div>`);
    }}

    function buildSummaryContent(row) {{
      const payload = row.payload || {{}};
      return sectionHtml('资产摘要', `
        <div class="metric-grid">
          ${{metricCard('期初持仓市值', formatMoney(payload.initial_holding_amount || 0))}}
          ${{metricCard('期初现金', formatMoney(payload.initial_cash || 0))}}
          ${{metricCard('期末持仓市值', formatMoney(payload.final_holding_amount || 0))}}
          ${{metricCard('期末现金', formatMoney(payload.final_cash || 0))}}
          ${{metricCard('期初总资产', formatMoney(payload.initial_total_asset || row.initial_total_asset))}}
          ${{metricCard('期末总资产', formatMoney(payload.final_total_asset || row.final_total_asset))}}
        </div>
      `);
    }}

    function buildDrilldownSections(row) {{
      const payload = row.payload || {{}};
      return {{
        initial_positions: {{ title: '期初持仓', html: recordsSection('期初持仓', payload.initial_positions || [], '当日开盘前没有持仓。', '持仓') }},
        buy_records: {{ title: '买入记录', html: recordsSection('买入记录', payload.buy_records || [], '当日没有买入记录。', '买入') }},
        sell_records: {{ title: '卖出记录', html: recordsSection('卖出记录', payload.sell_records || [], '当日没有卖出记录。', '卖出') }},
        final_positions: {{ title: '期末持仓', html: recordsSection('期末持仓', payload.final_positions || [], '当日收盘后没有持仓。', '收盘') }},
      }};
    }}

    function buildSectionButtons() {{
      return [
        {{ key: 'initial_positions', title: '期初持仓', desc: '查看开盘前已持有的仓位明细。' }},
        {{ key: 'buy_records', title: '买入', desc: '查看当日新开仓与加仓记录。' }},
        {{ key: 'sell_records', title: '卖出', desc: '查看当日止盈、止损和调仓卖出。' }},
        {{ key: 'final_positions', title: '期末持仓', desc: '查看收盘后剩余持仓状态。' }},
      ].map((item) => `
        <button class="panel section-button" type="button" data-section-key="${{item.key}}">
          <div class="section-button-title">${{item.title}}</div>
          <div class="section-button-desc">${{item.desc}}</div>
        </button>
      `).join('');
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

    function showSummaryPane() {{
      document.getElementById('sidebar-track').classList.remove('is-detail');
    }}

    function showSectionPane(title, html) {{
      document.getElementById('detail-panel-title').textContent = title;
      document.getElementById('detail-panel-content').innerHTML = html;
      document.getElementById('sidebar-track').classList.add('is-detail');
    }}

    function bindSectionButtons(row) {{
      const sections = buildDrilldownSections(row);
      document.querySelectorAll('[data-section-key]').forEach((btn) => {{
        btn.addEventListener('click', () => {{
          const key = btn.getAttribute('data-section-key');
          const section = sections[key];
          if (!section) return;
          showSectionPane(section.title, section.html);
        }});
      }});
    }}

    function updateDetail(idx) {{
      const row = rows[idx];
      document.getElementById('detail-date').textContent = row.date;
      document.getElementById('detail-asof').textContent = row.asof_date || '-';
      setValue('detail-daily-return', formatPct(row.daily_return), row.daily_return >= 0 ? 'positive' : 'negative');
      setValue('detail-cum-return', formatPct(row.cum_return), row.cum_return >= 0 ? 'positive' : 'negative');
      document.getElementById('detail-summary').innerHTML = buildSummaryContent(row);
      document.getElementById('section-buttons').innerHTML = buildSectionButtons();
      bindSectionButtons(row);
      showSummaryPane();
    }}

    function parseDate(value) {{
      return new Date(`${{value}}T00:00:00`);
    }}

    function getVisibleIndices(relayout, graphDiv = null) {{
      let start = 0;
      let end = rows.length - 1;
      const graphRange = graphDiv?.layout?.xaxis?.range || null;
      const x0 = relayout?.['xaxis.range[0]'] ?? relayout?.xaxis?.range?.[0] ?? graphRange?.[0];
      const x1 = relayout?.['xaxis.range[1]'] ?? relayout?.xaxis?.range?.[1] ?? graphRange?.[1];
      if (x0 || x1) {{
        const left = x0 ? parseDate(x0) : parseDate(rows[0].date);
        const right = x1 ? parseDate(x1) : parseDate(rows[rows.length - 1].date);
        while (start < rows.length - 1 && parseDate(rows[start].date) < left) start += 1;
        while (end > 0 && parseDate(rows[end].date) > right) end -= 1;
        if (end < start) return [0, rows.length - 1];
      }}
      return [start, end];
    }}

    function visibleRows(relayout, graphDiv = null) {{
      const [start, end] = getVisibleIndices(relayout, graphDiv);
      return rows.slice(start, end + 1);
    }}

    function computeYRange(relayout, mode, graphDiv = null) {{
      const currentRows = visibleRows(relayout, graphDiv);
      const values = [];
      if (mode === 'candlestick') {{
        currentRows.forEach((row) => {{
          values.push(row.start_cum_return * 100, row.cum_return * 100);
        }});
      }} else {{
        currentRows.forEach((row) => values.push(row.cum_return * 100));
      }}
      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = Math.max(max - min, 0.4);
      const pad = Math.max(span * 0.12, 0.08);
      return [min - pad, max + pad];
    }}

    function chooseMode(relayout, graphDiv = null) {{
      return visibleRows(relayout, graphDiv).length <= SHORT_WINDOW_THRESHOLD ? 'candlestick' : 'line';
    }}

    function buildTrace(mode) {{
      if (mode === 'candlestick') {{
        return {{
          type: 'candlestick',
          x: rows.map(r => r.date),
          open: rows.map(r => r.start_cum_return * 100),
          close: rows.map(r => r.cum_return * 100),
          high: rows.map(r => Math.max(r.start_cum_return, r.cum_return) * 100),
          low: rows.map(r => Math.min(r.start_cum_return, r.cum_return) * 100),
          customdata: rows.map((r, idx) => [idx, r.daily_return * 100, r.final_total_asset]),
          increasing: {{ line: {{ color: '#d92d20', width: 1.4 }}, fillcolor: '#d92d20' }},
          decreasing: {{ line: {{ color: '#1570ef', width: 1.4 }}, fillcolor: '#1570ef' }},
          whiskerwidth: 0,
          hovertemplate: [
            '日期: %{{x}}',
            '开盘累计收益率: %{{open:.2f}}%',
            '收盘累计收益率: %{{close:.2f}}%',
            '日收益率: %{{customdata[1]:.2f}}%',
            '期末总资产: %{{customdata[2]:,.2f}}',
            '<extra></extra>'
          ].join('<br>')
        }};
      }}
      return {{
        type: 'scatter',
        mode: 'lines',
        x: rows.map(r => r.date),
        y: rows.map(r => r.cum_return * 100),
        customdata: rows.map((r, idx) => [idx, r.daily_return * 100, r.final_total_asset]),
        line: {{ color: '#175cd3', width: 2.5, shape: 'spline', smoothing: 0.8 }},
        hovertemplate: [
          '日期: %{{x}}',
          '累计收益率: %{{y:.2f}}%',
          '日收益率: %{{customdata[1]:.2f}}%',
          '期末总资产: %{{customdata[2]:,.2f}}',
          '<extra></extra>'
        ].join('<br>')
      }};
    }}

    function buildLayout(mode, relayout, graphDiv = null) {{
      return {{
        margin: {{ l: 68, r: 24, t: 28, b: 64 }},
        hovermode: 'x unified',
        dragmode: 'zoom',
        showlegend: false,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {{
          title: '交易日',
          rangeslider: {{ visible: true }},
          showgrid: false,
          range: relayout?.['xaxis.range[0]'] ? [relayout['xaxis.range[0]'], relayout['xaxis.range[1]']] : undefined
        }},
        yaxis: {{
          title: mode === 'candlestick' ? '累计收益率 K 线 (%)' : '累计收益率 (%)',
          ticksuffix: '%',
          gridcolor: 'rgba(148, 163, 184, 0.18)',
          zerolinecolor: 'rgba(148, 163, 184, 0.25)',
          range: computeYRange(relayout, mode, graphDiv),
          fixedrange: false
        }}
      }};
    }}

    function renderChart(relayout = null) {{
      const graphDiv = document.getElementById('chart');
      const mode = chooseMode(relayout, graphDiv);
      const trace = buildTrace(mode);
      const layout = buildLayout(mode, relayout, graphDiv);
      return Plotly.react(graphDiv, [trace], layout, {{ responsive: true, displaylogo: false }});
    }}

    updateSummary();
    updateDetail(rows.length - 1);
    document.getElementById('detail-back').addEventListener('click', showSummaryPane);

    renderChart().then((plot) => {{
      plot.on('plotly_click', (event) => {{
        if (!event.points || !event.points.length) return;
        const idx = event.points[0].customdata?.[0];
        if (idx === undefined) return;
        updateDetail(idx);
      }});

      plot.on('plotly_relayout', (event) => {{
        if (syncLock) return;
        if (event['xaxis.autorange']) {{
          syncLock = true;
          renderChart(null).then(() => {{ syncLock = false; }});
          return;
        }}
        if (!event['xaxis.range[0]'] && !event['xaxis.range[1]']) return;
        syncLock = true;
        renderChart(event).then(() => {{ syncLock = false; }});
      }});
    }});
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
