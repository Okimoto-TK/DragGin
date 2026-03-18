from __future__ import annotations

import argparse
import json
import sys
import tempfile
import webbrowser
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from plotly.offline.offline import get_plotlyjs


UP_COLOR = "#D66A6A"   # muted red
DOWN_COLOR = "#5E9B6B" # muted green
NEUTRAL_LINE = "#6F8FB8"
ACCENT = "#6E8FD8"
BG = "#F5F7FB"
PANEL = "#FFFFFF"
CARD = "#F8FAFC"
BORDER = "#D8E0EC"
TEXT = "#1F2937"
TEXT_DIM = "#6B7280"


@dataclass
class DayRecord:
    trade_date: str
    prev_date: str
    open_asset: float
    close_asset: float
    high_asset: float
    low_asset: float
    daily_return: float
    cumulative_return: float
    initial_holding_amount: float
    initial_cash: float
    final_holding_amount: float
    final_cash: float
    initial_total_asset: float
    final_total_asset: float
    initial_positions: list[dict[str, Any]]
    sell_records: list[dict[str, Any]]
    buy_records: list[dict[str, Any]]
    final_positions: list[dict[str, Any]]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def load_backtest_records(backtest_dir: Path) -> list[DayRecord]:
    files = sorted(backtest_dir.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No json files found under: {backtest_dir}")

    raw: list[dict[str, Any]] = []
    for path in files:
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        if "date" not in payload:
            payload["date"] = path.stem
        raw.append(payload)

    if not raw:
        raise ValueError("No valid json payloads found.")

    raw.sort(key=lambda x: str(x.get("date", "")))
    first_initial = _safe_float(raw[0].get("initial_total_asset"), 0.0)
    if first_initial <= 0:
        raise ValueError("The first record has invalid initial_total_asset.")

    days: list[DayRecord] = []
    for item in raw:
        trade_date = str(item.get("date", ""))
        prev_date = str(item.get("asof_date", ""))
        open_asset = _safe_float(item.get("initial_total_asset"))
        close_asset = _safe_float(item.get("final_total_asset"))
        high_asset = max(open_asset, close_asset)
        low_asset = min(open_asset, close_asset)
        daily_return = (close_asset / open_asset - 1.0) if open_asset else 0.0
        cumulative_return = (close_asset / first_initial - 1.0) if first_initial else 0.0

        days.append(
            DayRecord(
                trade_date=trade_date,
                prev_date=prev_date,
                open_asset=open_asset,
                close_asset=close_asset,
                high_asset=high_asset,
                low_asset=low_asset,
                daily_return=daily_return,
                cumulative_return=cumulative_return,
                initial_holding_amount=_safe_float(item.get("initial_holding_amount")),
                initial_cash=_safe_float(item.get("initial_cash")),
                final_holding_amount=_safe_float(item.get("final_holding_amount")),
                final_cash=_safe_float(item.get("final_cash")),
                initial_total_asset=open_asset,
                final_total_asset=close_asset,
                initial_positions=list(item.get("initial_positions", []) or []),
                sell_records=list(item.get("sell_records", []) or []),
                buy_records=list(item.get("buy_records", []) or []),
                final_positions=list(item.get("final_positions", []) or []),
            )
        )

    return days


def build_html(days: list[DayRecord], title: str) -> str:
    plotly_js = get_plotlyjs()
    payload = json.dumps([asdict(x) for x in days], ensure_ascii=False)
    title_json = json.dumps(title, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: {BG};
      --panel: {PANEL};
      --card: {CARD};
      --border: {BORDER};
      --text: {TEXT};
      --text-dim: {TEXT_DIM};
      --accent: {ACCENT};
      --up: {UP_COLOR};
      --down: {DOWN_COLOR};
      --line: {NEUTRAL_LINE};
      --shadow: 0 18px 45px rgba(2, 8, 23, 0.34);
      --radius: 20px;
    }}

    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; height: 100%; background: linear-gradient(180deg, #f8fafc 0%, var(--bg) 100%); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
    body {{ overflow: hidden; }}

    .app {{ display: flex; flex-direction: column; height: 100vh; padding: 18px; gap: 14px; }}
    .header {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 8px 6px 2px; }}
    .title-wrap {{ display: flex; flex-direction: column; gap: 6px; }}
    .title {{ font-size: 22px; font-weight: 700; letter-spacing: 0.02em; }}
    .subtitle {{ color: var(--text-dim); font-size: 13px; }}
    .legend {{ display: flex; align-items: center; gap: 16px; color: var(--text-dim); font-size: 13px; flex-wrap: wrap; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 8px; }}
    .legend-dot {{ width: 10px; height: 10px; border-radius: 999px; }}

    .content {{ flex: 1; min-height: 0; display: grid; grid-template-columns: minmax(0, 1fr) 420px; gap: 16px; }}

    .chart-shell, .side-shell {{ min-height: 0; background: rgba(255, 255, 255, 0.92); border: 1px solid rgba(148, 163, 184, 0.22); box-shadow: 0 14px 34px rgba(148, 163, 184, 0.14); backdrop-filter: blur(10px); }}
    .chart-shell {{ border-radius: 24px; padding: 12px 14px 10px; display: flex; flex-direction: column; }}
    .side-shell {{ border-radius: 24px; overflow: hidden; position: relative; }}
    #chart {{ flex: 1; min-height: 0; }}

    .hint {{ padding: 0 10px 4px; color: var(--text-dim); font-size: 12px; }}

    .slider-wrap {{ width: 200%; height: 100%; display: flex; transform: translateX(0); transition: transform 260ms cubic-bezier(.22,.61,.36,1); }}
    .slider-wrap.detail-open {{ transform: translateX(-50%); }}
    .panel-page {{ width: 50%; height: 100%; padding: 16px; overflow: auto; }}
    .panel-page::-webkit-scrollbar {{ width: 10px; }}
    .panel-page::-webkit-scrollbar-thumb {{ background: rgba(148,163,184,.22); border-radius: 999px; }}
    .panel-page::-webkit-scrollbar-track {{ background: transparent; }}

    .section {{ background: linear-gradient(180deg, rgba(255,255,255,.98), rgba(248,250,252,.96)); border: 1px solid rgba(148,163,184,.16); border-radius: 18px; padding: 14px; margin-bottom: 14px; }}
    .section-title {{ font-size: 13px; letter-spacing: .08em; text-transform: uppercase; color: var(--text-dim); margin-bottom: 12px; font-weight: 700; }}

    .kv-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .kv-card {{ background: rgba(248, 250, 252, 0.98); border: 1px solid rgba(148,163,184,.14); border-radius: 16px; padding: 12px; min-width: 0; }}
    .kv-card.full {{ grid-column: 1 / -1; }}
    .kv-label {{ font-size: 12px; color: var(--text-dim); margin-bottom: 8px; }}
    .kv-value {{ font-size: 18px; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .pct-up {{ color: var(--up); }}
    .pct-down {{ color: var(--down); }}

    .buttons-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .action-btn {{ border: 1px solid rgba(110, 143, 216, .18); background: linear-gradient(180deg, rgba(255,255,255,.98), rgba(248,250,252,.96)); color: var(--text); border-radius: 16px; min-height: 78px; padding: 12px; text-align: left; cursor: pointer; transition: transform 140ms ease, border-color 140ms ease, background 140ms ease; }}
    .action-btn:hover {{ transform: translateY(-2px); border-color: rgba(110, 143, 216, .36); background: linear-gradient(180deg, rgba(255,255,255,.98), rgba(241,245,249,.98)); }}
    .action-btn .btn-title {{ display: block; font-size: 15px; font-weight: 700; margin-bottom: 7px; }}
    .action-btn .btn-sub {{ display: block; font-size: 12px; color: var(--text-dim); line-height: 1.45; }}

    .detail-top {{ display: flex; align-items: center; gap: 10px; margin-bottom: 12px; position: sticky; top: 0px; padding-bottom: 6px; background: linear-gradient(180deg, rgba(255,255,255,.98), rgba(255,255,255,.88), transparent); backdrop-filter: blur(8px); z-index: 3; }}
    .back-btn {{ border: 1px solid rgba(148,163,184,.18); background: rgba(248,250,252,.98); color: var(--text); border-radius: 12px; min-width: 38px; height: 38px; font-size: 20px; cursor: pointer; }}
    .detail-title {{ font-size: 18px; font-weight: 700; }}
    .detail-sub {{ font-size: 12px; color: var(--text-dim); margin-top: 2px; }}

    .table-wrap {{ border: 1px solid rgba(148,163,184,.14); border-radius: 16px; overflow: hidden; background: rgba(255,255,255,.96); }}
    table {{ width: 100%; border-collapse: collapse; table-layout: auto; }}
    thead {{ background: rgba(241,245,249,.98); position: sticky; top: 0; z-index: 2; }}
    th, td {{ padding: 10px 10px; border-bottom: 1px solid rgba(148,163,184,.08); font-size: 12.5px; text-align: right; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    th:first-child, td:first-child {{ text-align: left; }}
    tbody tr:hover {{ background: rgba(226,232,240,.45); }}
    .empty {{ padding: 28px 14px; text-align: center; color: var(--text-dim); font-size: 13px; }}

    @media (max-width: 1180px) {{
      .content {{ grid-template-columns: 1fr; }}
      .side-shell {{ height: min(52vh, 540px); }}
    }}
  </style>
  <script>{plotly_js}</script>
</head>
<body>
  <div class="app">
    <div class="header">
      <div class="title-wrap">
        <div class="title" id="page-title"></div>
        <div class="subtitle">横轴按交易日顺序显示，不补周末/节假日空白；主图支持点击选日，下方时间框可拖动选择范围。</div>
      </div>
      <div class="legend">
        <span class="legend-item"><span class="legend-dot" style="background: var(--up);"></span>上涨</span>
        <span class="legend-item"><span class="legend-dot" style="background: var(--down);"></span>下跌</span>
        <span class="legend-item"><span class="legend-dot" style="background: var(--line);"></span>大范围折线</span>
      </div>
    </div>

    <div class="content">
      <div class="chart-shell">
        <div class="hint" id="chart-hint"></div>
        <div id="chart"></div>
      </div>

      <div class="side-shell">
        <div class="slider-wrap" id="panel-slider">
          <div class="panel-page" id="overview-page">
            <div class="section">
              <div class="section-title">日信息概览</div>
              <div class="kv-grid" id="overview-grid"></div>
            </div>

            <div class="section">
              <div class="section-title">日资产摘要</div>
              <div class="kv-grid" id="asset-grid"></div>
            </div>

            <div class="section">
              <div class="section-title">明细</div>
              <div class="buttons-grid">
                <button class="action-btn" data-detail="initial_positions">
                  <span class="btn-title">期初持仓</span>
                  <span class="btn-sub">查看开盘前持仓列表与浮动盈亏。</span>
                </button>
                <button class="action-btn" data-detail="sell_records">
                  <span class="btn-title">卖出详细</span>
                  <span class="btn-sub">查看当日卖出价格、数量与实现盈亏。</span>
                </button>
                <button class="action-btn" data-detail="buy_records">
                  <span class="btn-title">买入详细</span>
                  <span class="btn-sub">查看当日买入成本与数量。</span>
                </button>
                <button class="action-btn" data-detail="final_positions">
                  <span class="btn-title">期末持仓</span>
                  <span class="btn-sub">查看收盘后持仓列表与浮动盈亏。</span>
                </button>
              </div>
            </div>
          </div>

          <div class="panel-page" id="detail-page">
            <div class="detail-top">
              <button class="back-btn" id="back-btn" title="返回">←</button>
              <div>
                <div class="detail-title" id="detail-title">明细</div>
                <div class="detail-sub" id="detail-sub">-</div>
              </div>
            </div>
            <div class="table-wrap" id="detail-wrap"></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const APP_TITLE = {title_json};
    const DAYS = {payload};
    const CANDLE_THRESHOLD = 90;
    const UP_COLOR = getComputedStyle(document.documentElement).getPropertyValue('--up').trim();
    const DOWN_COLOR = getComputedStyle(document.documentElement).getPropertyValue('--down').trim();
    const LINE_COLOR = getComputedStyle(document.documentElement).getPropertyValue('--line').trim();
    const ACCENT = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();

    const DAY_INDEX = DAYS.map((_, i) => i);

    let selectedIndex = Math.max(0, DAYS.length - 1);
    let currentRange = null;
    let currentMode = null;
    let currentDetail = 'final_positions';
    let gd = null;

    const DETAIL_META = {{
      initial_positions: {{ title: '期初持仓', desc: '期初持仓明细', columns: [
        {{ key: 'code', label: '代码', align: 'left' }},
        {{ key: 'cost', label: '成本' }},
        {{ key: 'qty', label: '数量' }},
        {{ key: 'float_pnl', label: '浮动盈亏', pnl: true }}
      ] }},
      sell_records: {{ title: '卖出详细', desc: '当日卖出成交', columns: [
        {{ key: 'code', label: '代码', align: 'left' }},
        {{ key: 'price', label: '卖价' }},
        {{ key: 'cost', label: '成本' }},
        {{ key: 'qty', label: '数量' }},
        {{ key: 'realized_pnl', label: '实现盈亏', pnl: true }}
      ] }},
      buy_records: {{ title: '买入详细', desc: '当日买入成交', columns: [
        {{ key: 'code', label: '代码', align: 'left' }},
        {{ key: 'cost', label: '成本' }},
        {{ key: 'qty', label: '数量' }}
      ] }},
      final_positions: {{ title: '期末持仓', desc: '期末持仓明细', columns: [
        {{ key: 'code', label: '代码', align: 'left' }},
        {{ key: 'cost', label: '成本' }},
        {{ key: 'qty', label: '数量' }},
        {{ key: 'float_pnl', label: '浮动盈亏', pnl: true }}
      ] }},
    }};

    function fmtNumber(value, digits = 2) {{
      const num = Number(value || 0);
      return num.toLocaleString('zh-CN', {{ minimumFractionDigits: digits, maximumFractionDigits: digits }});
    }}

    function fmtMoney(value) {{
      return fmtNumber(value, 2);
    }}

    function fmtQty(value) {{
      const num = Number(value || 0);
      const digits = Number.isInteger(num) ? 0 : 2;
      return num.toLocaleString('zh-CN', {{ minimumFractionDigits: digits, maximumFractionDigits: digits }});
    }}

    function fmtPct(value) {{
      const num = Number(value || 0) * 100;
      return `${{num >= 0 ? '+' : ''}}${{num.toFixed(2)}}%`;
    }}

    function pnlClass(value) {{
      return Number(value || 0) >= 0 ? 'pct-up' : 'pct-down';
    }}

    function clampIndex(x) {{
      return Math.max(0, Math.min(DAYS.length - 1, Number(x)));
    }}

    function normalizeRange(range) {{
      if (!range || range.length < 2) return null;
      return [clampIndex(range[0]), clampIndex(range[1])];
    }}

    function isInRange(idx, range) {{
      if (!range || range[0] == null || range[1] == null) return true;
      return idx >= range[0] && idx <= range[1];
    }}

    function visibleCount() {{
      return DAY_INDEX.filter(i => isInRange(i, currentRange)).length;
    }}

    function modeForRange() {{
      return visibleCount() > CANDLE_THRESHOLD ? 'line' : 'candlestick';
    }}

    function selectedDay() {{
      return DAYS[Math.max(0, Math.min(selectedIndex, DAYS.length - 1))];
    }}

    function closestIndexByX(x) {{
      return Math.round(clampIndex(x));
    }}

    function tickSpec() {{
      const range = normalizeRange(currentRange) || [0, DAYS.length - 1];
      const left = Math.max(0, Math.floor(range[0]));
      const right = Math.min(DAYS.length - 1, Math.ceil(range[1]));
      const span = Math.max(1, right - left + 1);
      const targetTicks = 8;
      const step = Math.max(1, Math.ceil(span / targetTicks));
      const tickvals = [];
      const ticktext = [];
      for (let i = left; i <= right; i += step) {{
        tickvals.push(i);
        ticktext.push(DAYS[i].trade_date);
      }}
      if (tickvals[tickvals.length - 1] !== right) {{
        tickvals.push(right);
        ticktext.push(DAYS[right].trade_date);
      }}
      return {{ tickvals, ticktext, range: [left - 0.5, right + 0.5], left, right }};
    }}

    function visibleDays() {{
      const ticks = tickSpec();
      return DAYS.slice(ticks.left, ticks.right + 1);
    }}

    function canUseLogAxis() {{
      return visibleDays().every(d =>
        Number(d.open_asset) > 0 &&
        Number(d.close_asset) > 0 &&
        Number(d.high_asset) > 0 &&
        Number(d.low_asset) > 0
      );
    }}

    function computeYAxisConfig() {{
      const vals = visibleDays().flatMap(d => [
        Number(d.open_asset || 0),
        Number(d.close_asset || 0),
        Number(d.high_asset || 0),
        Number(d.low_asset || 0),
      ]).filter(v => Number.isFinite(v) && v > 0);

      if (!vals.length) {{
        return {{ type: 'linear' }};
      }}

      const minVal = Math.min(...vals);
      const maxVal = Math.max(...vals);

      if (!canUseLogAxis()) {{
        const span = Math.max(maxVal - minVal, maxVal * 0.01, 1e-9);
        const pad = Math.max(span * 0.012, maxVal * 0.0015);
        return {{
          type: 'linear',
          range: [Math.max(0, minVal - pad), maxVal + pad]
        }};
      }}

      const safeMin = Math.max(minVal, 1e-9);
      const safeMax = Math.max(maxVal, safeMin * 1.000001);
      const logMin = Math.log10(safeMin);
      const logMax = Math.log10(safeMax);
      const logSpan = Math.max(logMax - logMin, 0.0025);
      const logPad = Math.max(logSpan * 0.008, 0.0018);
      return {{
        type: 'log',
        range: [logMin - logPad, logMax + logPad]
      }};
    }}

    function renderInfoCards() {{
      const d = selectedDay();

      const overview = [
        {{ label: '交易日', value: d.trade_date }},
        {{ label: '上交易日', value: d.prev_date || '-' }},
        {{ label: '日收益率', value: fmtPct(d.daily_return), cls: pnlClass(d.daily_return) }},
        {{ label: '累计收益率', value: fmtPct(d.cumulative_return), cls: pnlClass(d.cumulative_return) }},
      ];

      const assets = [
        {{ label: '期初持仓市值', value: fmtMoney(d.initial_holding_amount) }},
        {{ label: '期末持仓市值', value: fmtMoney(d.final_holding_amount) }},
        {{ label: '期初现金', value: fmtMoney(d.initial_cash) }},
        {{ label: '期末现金', value: fmtMoney(d.final_cash) }},
        {{ label: '期初总资产', value: fmtMoney(d.initial_total_asset)}},
        {{ label: '期末总资产', value: fmtMoney(d.final_total_asset)}},
      ];

      document.getElementById('overview-grid').innerHTML = overview.map(item => `
        <div class="kv-card">
          <div class="kv-label">${{item.label}}</div>
          <div class="kv-value ${{item.cls || ''}}">${{item.value}}</div>
        </div>
      `).join('');

      document.getElementById('asset-grid').innerHTML = assets.map(item => `
        <div class="kv-card ${{item.full ? 'full' : ''}}">
          <div class="kv-label">${{item.label}}</div>
          <div class="kv-value">${{item.value}}</div>
        </div>
      `).join('');
    }}

    function cellHtml(value, col) {{
      if (col.key === 'qty') return fmtQty(value);
      if (typeof value === 'number') return fmtMoney(value);
      return value == null ? '-' : String(value);
    }}

    function renderDetailTable() {{
      const d = selectedDay();
      const meta = DETAIL_META[currentDetail];
      const rows = d[currentDetail] || [];
      document.getElementById('detail-title').textContent = meta.title;
      document.getElementById('detail-sub').textContent = `${{d.trade_date}} · ${{meta.desc}} · 共 ${{rows.length}} 条`;

      const wrap = document.getElementById('detail-wrap');
      if (!rows.length) {{
        wrap.innerHTML = '<div class="empty">该日没有对应明细。</div>';
        return;
      }}

      const thead = `<thead><tr>${{meta.columns.map(col => `<th style="text-align:${{col.align || 'right'}}">${{col.label}}</th>`).join('')}}</tr></thead>`;
      const tbody = `<tbody>${{rows.map(row => `<tr>${{meta.columns.map(col => {{
        const raw = row[col.key];
        const cls = col.pnl ? pnlClass(raw) : '';
        const align = col.align || 'right';
        return `<td class="${{cls}}" style="text-align:${{align}}">${{cellHtml(raw, col)}}</td>`;
      }}).join('')}}</tr>`).join('')}}</tbody>`;
      wrap.innerHTML = `<table>${{thead}}${{tbody}}</table>`;
    }}

    function detailOpen(open) {{
      document.getElementById('panel-slider').classList.toggle('detail-open', open);
    }}

    function buildChartTraces(mode) {{
      const x = DAY_INDEX;
      const tradeDates = DAYS.map(d => d.trade_date);
      const open = DAYS.map(d => d.open_asset);
      const high = DAYS.map(d => d.high_asset);
      const low = DAYS.map(d => d.low_asset);
      const close = DAYS.map(d => d.close_asset);
      const traces = [];

      if (mode === 'line') {{
        traces.push({{
          type: 'scatter',
          mode: 'lines',
          x, y: close, customdata: tradeDates,
          line: {{ color: LINE_COLOR, width: 2.8 }},
          hovertemplate: '交易日: %{{customdata}}<br>期末总资产: %{{y:,.3~s}}<extra></extra>',
          name: '总资产'
        }});
      }} else {{
        traces.push({{
          type: 'candlestick',
          x, open, high, low, close, customdata: tradeDates,
          whiskerwidth: 0,
          increasing: {{ line: {{ color: UP_COLOR, width: 1.2 }}, fillcolor: UP_COLOR }},
          decreasing: {{ line: {{ color: DOWN_COLOR, width: 1.2 }}, fillcolor: DOWN_COLOR }},
          hovertemplate: '交易日: %{{customdata}}<br>期初总资产: %{{open:,.3~s}}<br>期末总资产: %{{close:,.3~s}}<extra></extra>',
          name: '日资产K'
        }});
      }}

      const sel = selectedDay();
      traces.push({{
        type: 'scatter', mode: 'markers',
        x: [selectedIndex], y: [sel.close_asset],
        marker: {{ size: 13, color: 'rgba(255,255,255,.96)', line: {{ width: 2, color: ACCENT }} }},
        hoverinfo: 'skip', showlegend: false,
        name: '选中'
      }});
      return traces;
    }}

    function buildChartLayout(mode) {{
      const sel = selectedDay();
      const ticks = tickSpec();
      const yAxisCfg = computeYAxisConfig();
      return {{
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: {{ l: 58, r: 22, t: 18, b: 50 }},
        showlegend: false,
        hovermode: 'x unified',
        dragmode: 'zoom',
        xaxis: {{
          type: 'linear',
          showgrid: false,
          zeroline: false,
          tickfont: {{ color: '#6B7280' }},
          tickmode: 'array',
          tickvals: ticks.tickvals,
          ticktext: ticks.ticktext,
          showspikes: true,
          spikecolor: 'rgba(148,163,184,.45)',
          spikethickness: 1,
          spikesnap: 'cursor',
          rangeslider: {{
            visible: true,
            thickness: 0.12,
            bgcolor: 'rgba(241,245,249,.96)',
            bordercolor: 'rgba(148,163,184,.22)',
            borderwidth: 1
          }},
          range: currentRange || ticks.range,
        }},
        yaxis: {{
          title: {{ text: '总资产 (M)', font: {{ color: '#6B7280', size: 12 }} }},
          type: yAxisCfg.type,
          range: yAxisCfg.range,
          autorange: false,
          tickfont: {{ color: '#6B7280' }},
          gridcolor: 'rgba(148,163,184,.18)',
          zeroline: false,
          separatethousands: true,
          tickformat: '~s',
          minexponent: 6
        }},
        shapes: [{{
          type: 'line', xref: 'x', yref: 'paper',
          x0: selectedIndex, x1: selectedIndex,
          y0: 0, y1: 1,
          line: {{ color: 'rgba(110,143,216,.60)', width: 1.25, dash: 'dot' }}
        }}],
        annotations: [{{
          x: selectedIndex, y: Math.max(Number(sel.close_asset || 0), 1e-9),
          xref: 'x', yref: 'y',
          text: `选中: ${{sel.trade_date}}`,
          showarrow: true,
          arrowhead: 2,
          ax: 40,
          ay: -40,
          font: {{ size: 11, color: '#1F2937' }},
          bgcolor: 'rgba(255,255,255,.98)',
          bordercolor: 'rgba(110,143,216,.34)',
          borderwidth: 1,
          borderpad: 6,
          arrowcolor: 'rgba(110,143,216,.58)'
        }}]
      }};
    }}

    function renderChart(force = false) {{
      const mode = modeForRange();
      document.getElementById('chart-hint').textContent = mode === 'line'
        ? `当前窗口共 ${{visibleCount()}} 个交易日，已切换为折线视图。`
        : `当前窗口共 ${{visibleCount()}} 个交易日，使用无影线日K视图。`;

      const traces = buildChartTraces(mode);
      const layout = buildChartLayout(mode);
      const config = {{ responsive: true, displaylogo: false, scrollZoom: true }};
      Plotly.react('chart', traces, layout, config).then(() => {{
        gd = document.getElementById('chart');
        if (force || currentMode == null) bindChartEvents();
      }});
      currentMode = mode;
    }}

    function bindChartEvents() {{
      if (!gd || gd.__eventsBound) return;
      gd.__eventsBound = true;

      gd.on('plotly_click', (ev) => {{
        if (!ev || !ev.points || !ev.points.length) return;
        const p = ev.points[0];
        if (typeof p.pointIndex === 'number') {{
          selectedIndex = p.pointIndex;
        }} else if (p.x != null) {{
          selectedIndex = closestIndexByX(p.x);
        }}
        renderInfoCards();
        renderDetailTable();
        renderChart(false);
      }});

      gd.on('plotly_relayout', (ev) => {{
        if (!ev) return;
        let nextRange = currentRange;
        if (Array.isArray(ev['xaxis.range'])) {{
          nextRange = ev['xaxis.range'];
        }} else if (ev['xaxis.range[0]'] && ev['xaxis.range[1]']) {{
          nextRange = [ev['xaxis.range[0]'], ev['xaxis.range[1]']];
        }} else if (ev['xaxis.autorange']) {{
          nextRange = null;
        }} else {{
          return;
        }}

        const prevMode = currentMode;
        currentRange = normalizeRange(nextRange);
        const nextMode = modeForRange();
        if (prevMode !== nextMode) {{
          renderChart(false);
        }}
      }});
    }}

    function setupButtons() {{
      document.querySelectorAll('.action-btn').forEach(btn => {{
        btn.addEventListener('click', () => {{
          currentDetail = btn.dataset.detail;
          renderDetailTable();
          detailOpen(true);
        }});
      }});
      document.getElementById('back-btn').addEventListener('click', () => detailOpen(false));
    }}

    function boot() {{
      document.getElementById('page-title').textContent = APP_TITLE;
      renderInfoCards();
      renderDetailTable();
      renderChart(true);
      setupButtons();
    }}

    boot();
  </script>
</body>
</html>
"""


def write_html_file(html: str, output_html: Path | None) -> Path:
    if output_html is not None:
        output_html.parent.mkdir(parents=True, exist_ok=True)
        output_html.write_text(html, encoding="utf-8")
        return output_html

    tmp_dir = Path(tempfile.mkdtemp(prefix="backtest_viz_"))
    out = tmp_dir / "backtest_visualizer.html"
    out.write_text(html, encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest visualizer")
    parser.add_argument("--backtest-dir", required=True, help="Directory containing daily json files")
    parser.add_argument("--title", default="Backtest Visualizer", help="Page title")
    parser.add_argument("--output-html", default="", help="Optional path to save the generated html")
    parser.add_argument("--no-open", action="store_true", help="Generate html but do not open browser")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backtest_dir = Path(args.backtest_dir).expanduser().resolve()
    if not backtest_dir.exists() or not backtest_dir.is_dir():
        print(f"[ERROR] backtest-dir does not exist or is not a directory: {backtest_dir}", file=sys.stderr)
        return 2

    try:
        days = load_backtest_records(backtest_dir)
        html = build_html(days, args.title)
        output_path = write_html_file(html, Path(args.output_html).expanduser().resolve() if args.output_html else None)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"Generated HTML: {output_path}")
    if not args.no_open:
        webbrowser.open(output_path.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
