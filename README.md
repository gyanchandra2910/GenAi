<div align="center">

# 🇮🇳 AI for the Indian Investor

### NSE Intelligence Platform — Full-Stack AI Advisory Application

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.3+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-Llama--3.3--70b-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> **Team: The Silicon Savants** | ET Gen AI Hackathon 2026

*Turning raw NSE market data into institutional-grade investment insights — in seconds.*

</div>

---

## 📌 Overview

**AI for the Indian Investor** is a full-stack, dual-agent intelligence platform that processes real-time NSE stock data and company news using large language models to surface **actionable investment signals** — the kind previously available only to institutional traders.

The platform runs two specialised AI agents in parallel and presents results through a **sleek dark-themed Streamlit dashboard**:

| Agent | Input | Output |
|-------|-------|--------|
| 🔭 **Opportunity Radar** | Company news & filings | Fundamental signals with confidence scores |
| 📊 **Chart Intelligence** | OHLCV price history | Technical patterns + plain-English LLM narrative |

---

## ✨ Features

### 🤖 AI Intelligence Layer
- **⚡ Ultra-Low Latency LLM** — Groq inference (Llama-3.3-70b-versatile) delivers sub-second AI responses
- **🔭 Opportunity Radar Agent** — Scans Yahoo Finance news for 9 signal types (insider buys, bulk deals, earnings beats, fundraising, and more) with LLM-assigned confidence scores
- **📊 Chart Intelligence Agent** — Computes RSI-14, SMA-20/50, Bollinger Bands, and volume spikes; narrates the full technical setup in plain English
- **📡 Real-Time NSE Data** — Live OHLCV data and news via `yfinance` for any NSE-listed equity

### 🎨 Streamlit Frontend (`app.py`)
- **🌑 Dark Premium Theme** — Custom CSS with navy/orange palette, glassmorphism-style signal cards
- **⚡ Dual-Column Dashboard** — Fundamental signals and technical signals displayed side-by-side
- **🏷️ Confidence Badges** — Colour-coded 🟢 High / 🟡 Medium / ⚪ Low confidence indicators per signal
- **🧭 Direction Chips** — ▲ BULLISH / ▼ BEARISH / ● NEUTRAL labels per technical pattern
- **💬 LLM Narrative Block** — Plain-English 3–5 sentence technical briefing from Groq
- **🗃️ Market Data Preview** — First 5 OHLCV candles in an interactive collapsible table
- **❌ Graceful Error Handling** — Connection refused, timeouts, and HTTP errors surface as friendly messages

### 🏗️ Backend Architecture
- **🛡️ Fault-Tolerant Pipeline** — Parallel `asyncio.gather` execution; if one agent fails, the other still returns results (`partial` status)
- **📋 Type-Safe API** — Full Pydantic v2 request/response validation with auto-generated Swagger docs
- **🔧 12-Factor Config** — All secrets via environment variables; zero hardcoded credentials

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI / Frontend** | Streamlit 1.3+ (dark theme, custom CSS) |
| **Web Framework** | FastAPI + Uvicorn (ASGI) |
| **LLM Runtime** | Groq — `llama-3.3-70b-versatile` |
| **AI Orchestration** | LangChain (LCEL chains) |
| **Market Data** | yfinance (NSE via `.NS` suffix) |
| **Technical Analysis** | `ta` library (RSI, SMA, Bollinger Bands) |
| **Data Processing** | Pandas + NumPy |
| **Config Management** | Pydantic-Settings + python-dotenv |
| **Language** | Python 3.11+ |

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/gyanchandra2910/GenAi.git
cd GenAi
```

### 2. Create & Activate a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install streamlit
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and set your Groq API key:

```env
GROQ_API_KEY=gsk_your_actual_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
APP_ENV=development
```

> 🔑 Get a free Groq API key at [console.groq.com](https://console.groq.com)

---

## ▶️ Running the Application (Two Terminals)

The platform requires **two services running simultaneously**.

### Terminal 1 — FastAPI Backend

```bash
# Activate venv first
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

# Start the backend API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend is live at **`http://localhost:8000`**

| Endpoint | Description |
|----------|-------------|
| `GET  /health` | Liveness check |
| `POST /api/v1/analyze` | Full dual-agent stock analysis |
| `GET  /docs` | Interactive Swagger UI |

---

### Terminal 2 — Streamlit Frontend

```bash
# In a NEW terminal window, activate venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

# Start the frontend dashboard
streamlit run app.py
```

Frontend is live at **`http://localhost:8501`**

> ⚠️ **Important:** Start the FastAPI backend (Terminal 1) **before** opening the Streamlit UI.

---

## 📡 REST API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Full Stock Analysis — `POST /api/v1/analyze`

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "TCS",
    "period": "3mo",
    "interval": "1d"
  }'
```

**Example Response:**

```json
{
  "status": "success",
  "ticker": "TCS.NS",
  "message": "Analysed TCS.NS | 63 candles | 2 fundamental signal(s) | 3 technical signal(s).",
  "fundamental_signals": [
    {
      "ticker": "TCS",
      "signal_type": "PARTNERSHIP",
      "confidence": 0.82,
      "summary": "TCS signed a multi-year digital transformation deal..."
    }
  ],
  "technical_signals": [
    {
      "pattern_name": "Price Above Both SMAs (Uptrend)",
      "direction": "BULLISH",
      "confidence": 0.65,
      "narrative": "TCS is trading above its 20-day (₹3,421) and 50-day (₹3,387) moving averages..."
    }
  ]
}
```

### Supported Tickers (Examples)

```
TCS · RELIANCE · INFY · HDFCBANK · WIPRO · BAJFINANCE · ADANIENT · TATAMOTORS
```

---

## 📁 Project Structure

```
.
├── app.py                       # Streamlit frontend (dark theme dashboard)
├── main.py                      # FastAPI app factory + health check
├── requirements.txt             # Pinned dependencies
├── .env.example                 # Environment variable template
│
├── api/
│   └── v1/
│       └── router.py            # POST /analyze endpoint (dual-agent pipeline)
│
├── agents/
│   ├── opportunity_radar.py     # Fundamental signals agent (LangChain + Groq)
│   └── chart_intelligence.py   # Technical analysis agent (ta + LangChain + Groq)
│
├── services/
│   └── market_data.py           # yfinance data fetching (OHLCV + news)
│
├── models/
│   └── schemas.py               # Pydantic request/response schemas
│
├── core/
│   └── config.py                # Pydantic-Settings environment management
│
└── utils/                       # Shared helper functions (future use)
```

---

## 🔮 Roadmap

- [ ] WebSocket endpoint for live tick-by-tick streaming
- [ ] Portfolio-level analysis across multiple tickers
- [ ] Backtesting engine for historical signal validation
- [ ] WhatsApp / Telegram alert integration via Twilio
- [ ] Candlestick charts with pattern overlays (Plotly)

---

## 🤝 Team

**The Silicon Savants** — ET Gen AI Hackathon 2026

---

<div align="center">

Built with ❤️ for the Indian retail investor.

*"The stock market is a device for transferring money from the impatient to the patient." — Warren Buffett*

</div>
