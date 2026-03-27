# AI for the Indian Investor

**ET Gen AI Hackathon 2026 | Problem Statement 6: AI for the Indian Investor**

Institutional-grade stock intelligence for Indian retail investors.

Built by The Silicon Savants for ET Gen AI Hackathon 2026.

## 1. Project Title & Tagline

AI for the Indian Investor

Institutional-grade stock market analysis for retail investors, built to reduce information asymmetry and guard against HFT-driven traps.

## 2. The Problem & Solution

### Problem

India has 14 crore+ demat accounts, but most retail investors are flying blind — reacting to tips, missing filings, unable to read technicals, and managing mutual fund portfolios on gut feel. ET Markets has the data. Build the intelligence layer that turns data into actionable, money-making decisions.

### Solution

This project provides a dual-agent AI analysis pipeline that merges:

- Fundamental signal extraction from company news and events.
- Technical signal detection from price-action indicators.

The result is a single, low-latency analysis response delivered through FastAPI and Streamlit, so investors can make informed decisions in seconds rather than hours.

## 3. Impact Model (Quantified)

Back-of-the-envelope impact estimate:

| Metric | Manual Workflow | AI Workflow | Improvement |
| --- | --- | --- | --- |
| Research time per stock | 3-5 hours | < 5 seconds | ~99.9% time saved |

Business and investor impact:

- Reduces dependence on premium advisory products that often cost ₹5,000+ per month.
- Delivers a democratized API-driven intelligence layer for retail participants.
- Protects capital by reducing emotional, reactionary trading behavior.

## 4. Architecture & Agent Roles

### System Architecture

- Frontend: Streamlit dashboard for user input and signal display.
- Backend: FastAPI orchestration layer with typed request/response models.
- Data: yfinance for market data and company news.
- Intelligence: Groq-hosted Llama-3.3-70B via LangChain workflows.

### Agent 1: Opportunity Radar (Fundamental Agent)

- Scans recent company news and event context.
- Extracts fundamental signals and confidence scores.
- Converts unstructured headlines into decision-ready summaries.

### Agent 2: Chart Intelligence (Technical Agent)

- Computes technical indicators including RSI, SMA, and Bollinger Bands.
- Detects chart conditions and directional setup.
- Produces concise technical narrative for the current market state.

### Communication Model

Both agents execute in parallel using Python `asyncio.gather` to minimize end-to-end latency.

### Error Handling Model

The pipeline uses graceful degradation: if one agent fails to fetch or process data, the API returns partial success with available outputs from the healthy agent instead of failing the full request.

## 5. Tech Stack

- Streamlit
- FastAPI
- Groq (Llama-3.3)
- yfinance
- LangChain
- pandas
- ta

## 6. Setup & Run Instructions

Run the backend and frontend in two terminals.

1. Terminal 1 (API)

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. Terminal 2 (UI)

```bash
streamlit run app.py
```

## 7. Footer

*Built by The Silicon Savants for the ET Gen AI Hackathon 2026.*
