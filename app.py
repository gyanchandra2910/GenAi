"""
app.py — Streamlit Frontend for AI for the Indian Investor
===========================================================
Run with: streamlit run app.py
"""

import pandas as pd
import requests
import streamlit as st

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI for the Indian Investor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (dark premium theme) ──────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #0f1923; color: #e0e6f0; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background-color: #1a2840; }

  /* Cards */
  .signal-card {
    background: #1e2d45;
    border-left: 4px solid #e8721c;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
  }
  .signal-card.technical { border-left-color: #1a9e6e; }
  .signal-card h4 { margin: 0 0 6px 0; color: #f0f4ff; font-size: 0.95rem; }
  .signal-card p  { margin: 0; color: #b0bcd0; font-size: 0.88rem; line-height: 1.5; }

  /* Confidence badge */
  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-top: 8px;
  }
  .badge-high   { background:#1a9e6e22; color:#1a9e6e; border:1px solid #1a9e6e55; }
  .badge-medium { background:#e8721c22; color:#e8721c; border:1px solid #e8721c55; }
  .badge-low    { background:#6b748533; color:#8090a8; border:1px solid #6b748555; }

  /* Direction chip */
  .chip {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 10px;
    font-size: 0.74rem;
    font-weight: 600;
    margin-left: 8px;
  }
  .chip-bull { background:#1a9e6e33; color:#1a9e6e; }
  .chip-bear { background:#e8303033; color:#e83030; }
  .chip-neu  { background:#6b748533; color:#8090a8; }

  /* Divider */
  hr { border-color: #2a3d55; }

  /* Metric override */
  div[data-testid="metric-container"] {
    background: #1e2d45;
    border-radius: 8px;
    padding: 12px 16px;
  }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000/api/v1/analyze"

SIGNAL_EMOJI = {
    "INSIDER_BUY":    "🏦",
    "BULK_DEAL":      "📦",
    "EARNINGS_BEAT":  "📊",
    "FUNDRAISE":      "💰",
    "BOARD_CHANGE":   "👔",
    "PARTNERSHIP":    "🤝",
    "REGULATORY_WIN": "✅",
    "BEARISH_FLAG":   "🚩",
    "NEUTRAL":        "➖",
}


def confidence_badge(score: float) -> str:
    pct = int(score * 100)
    if score >= 0.7:
        return f'<span class="badge badge-high">🟢 {pct}% confidence</span>'
    elif score >= 0.5:
        return f'<span class="badge badge-medium">🟡 {pct}% confidence</span>'
    else:
        return f'<span class="badge badge-low">⚪ {pct}% confidence</span>'


def direction_chip(direction: str) -> str:
    if direction == "BULLISH":
        return '<span class="chip chip-bull">▲ BULLISH</span>'
    elif direction == "BEARISH":
        return '<span class="chip chip-bear">▼ BEARISH</span>'
    return '<span class="chip chip-neu">● NEUTRAL</span>'


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Analysis Settings")
    st.markdown("---")

    ticker = st.text_input(
        "🏷️ NSE Ticker Symbol",
        value="RELIANCE",
        max_chars=20,
        help="Enter the NSE symbol without .NS suffix, e.g. TCS, INFY, HDFCBANK",
    ).strip().upper()

    period = st.selectbox(
        "📅 Historical Period",
        options=["1mo", "3mo", "6mo", "1y"],
        index=1,
        help="Look-back window for OHLCV data and pattern detection",
    )

    st.markdown("---")
    run_btn = st.button("🔍 Analyze Stock", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown(
        "<small style='color:#6b7485'>Powered by **Groq** · llama-3.3-70b-versatile<br>"
        "Data via **yfinance** (NSE)</small>",
        unsafe_allow_html=True,
    )

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; padding: 10px 0 4px 0;">
      <h1 style="color:#e8721c; font-size:2.1rem; margin-bottom:2px;">
        📈 AI for the Indian Investor
      </h1>
      <p style="color:#7080a0; font-size:1rem; margin:0;">
        NSE Intelligence Platform &nbsp;·&nbsp; Dual-Agent AI Advisory &nbsp;·&nbsp;
        <strong style="color:#e8721c">The Silicon Savants</strong>
      </p>
    </div>
    <hr>
    """,
    unsafe_allow_html=True,
)

# ─── Default state ────────────────────────────────────────────────────────────
if not run_btn:
    col1, col2, col3 = st.columns(3)
    col1.metric("🔭 Opportunity Radar", "9 Signal Types", "Fundamental AI")
    col2.metric("📊 Chart Intelligence", "5 Pattern Rules", "Technical AI")
    col3.metric("⚡ LLM Inference", "< 1 sec", "Groq LPU")

    st.markdown("""
    <div style="color:#5a6880; text-align:center; margin-top:40px; font-size:0.9rem;">
      Enter a ticker in the sidebar and click <strong>Analyze Stock</strong> to get started.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── API Call ─────────────────────────────────────────────────────────────────
with st.spinner(f"🤖 AI is analyzing **{ticker}** — fetching NSE data & running agents..."):
    try:
        resp = requests.post(
            API_URL,
            json={"ticker": ticker, "period": period},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ **Cannot connect to the backend.**  \n"
            "Make sure FastAPI is running:  \n"
            "```\n.venv\\Scripts\\python.exe -m uvicorn main:app --reload\n```"
        )
        st.stop()

    except requests.exceptions.Timeout:
        st.error("⏱️ **Request timed out** (>60 s). The LLM may be slow — try again.")
        st.stop()

    except requests.exceptions.HTTPError as e:
        st.error(f"🚫 **HTTP {resp.status_code}:** {resp.text[:300]}")
        st.stop()

    except Exception as e:
        st.error(f"⚠️ Unexpected error: {e}")
        st.stop()

# ─── Status Banner ────────────────────────────────────────────────────────────
status = data.get("status", "unknown")
message = data.get("message", "")
ticker_ns = data.get("ticker", ticker)

status_colour = {"success": "#1a9e6e", "partial": "#e8721c", "error": "#e83030"}.get(status, "#6b7485")
status_icon   = {"success": "✅", "partial": "⚡", "error": "❌"}.get(status, "ℹ️")

st.markdown(
    f"""
    <div style="background:#1e2d45; border-radius:10px; padding:14px 20px; margin-bottom:18px;
                border-left:4px solid {status_colour};">
      <span style="font-size:1.05rem; color:#e0e6f0;">
        {status_icon} <strong>{ticker_ns}</strong> &nbsp;·&nbsp;
        <span style="color:{status_colour}; text-transform:uppercase; font-size:0.8rem;">{status}</span>
      </span><br>
      <span style="color:#7080a0; font-size:0.85rem;">{message}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Data Preview ─────────────────────────────────────────────────────────────
preview = data.get("data_preview")
if preview:
    with st.expander("🗃️ Market Data Preview", expanded=False):
        df = pd.DataFrame(preview.get("preview_rows", []))
        st.dataframe(df, use_container_width=True)
        st.caption(
            f"Showing first {len(df)} of {preview['total_rows']} candles · "
            f"Period: {preview['period']} · Interval: {preview['interval']}"
        )

st.markdown("---")

# ─── Two-column agent results ─────────────────────────────────────────────────
col_fund, col_tech = st.columns(2, gap="large")

# ── Fundamental Signals ───────────────────────────────────────────────────────
with col_fund:
    st.markdown("### 🔭 Opportunity Radar")
    st.caption("Fundamental signals from corporate news & filings")

    fund_err = data.get("fundamental_error")
    fund_sigs = data.get("fundamental_signals", [])

    if fund_err:
        st.warning(f"⚠️ Agent error: {fund_err}")
    elif not fund_sigs:
        st.info("No fundamental signals detected for this period.")
    else:
        for sig in fund_sigs:
            emoji = SIGNAL_EMOJI.get(sig.get("signal_type", "NEUTRAL"), "🔹")
            st.markdown(
                f"""
                <div class="signal-card">
                  <h4>{emoji} {sig.get('signal_type','').replace('_',' ').title()}</h4>
                  <p>{sig.get('summary','')}</p>
                  {confidence_badge(sig.get('confidence', 0))}
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── Technical Signals ─────────────────────────────────────────────────────────
with col_tech:
    st.markdown("### 📊 Chart Intelligence")
    st.caption("Technical patterns: RSI, SMA, Bollinger Bands, Volume")

    tech_err = data.get("technical_error")
    tech_sigs = data.get("technical_signals", [])

    if tech_err:
        st.warning(f"⚠️ Agent error: {tech_err}")
    elif not tech_sigs:
        st.info("No technical signals detected for this period.")
    else:
        # Show the LLM narrative from the top signal
        top_narrative = tech_sigs[0].get("narrative", "")
        if top_narrative:
            st.markdown(
                f"""
                <div style="background:#162235; border-radius:8px; padding:14px 16px;
                            border:1px solid #2a3d55; margin-bottom:14px; font-size:0.9rem;
                            color:#b8c8e0; line-height:1.6;">
                  💬 <em>{top_narrative}</em>
                </div>
                """,
                unsafe_allow_html=True,
            )

        for pat in tech_sigs:
            direction = pat.get("direction", "NEUTRAL")
            st.markdown(
                f"""
                <div class="signal-card technical">
                  <h4>
                    {pat.get('pattern_name','')}
                    {direction_chip(direction)}
                  </h4>
                  {confidence_badge(pat.get('confidence', 0))}
                </div>
                """,
                unsafe_allow_html=True,
            )

# ─── Raw JSON Expander ────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🔩 Raw API Response (JSON)", expanded=False):
    st.json(data)
