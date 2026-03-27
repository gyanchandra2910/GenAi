"""
api/v1/router.py — API Version 1 Router
=========================================
Resilient dual-agent /analyze endpoint for the NSE Intelligence Platform.

Pipeline per request
--------------------
1. Fetch OHLCV data + company news in parallel (asyncio.gather).
2. Run OpportunityRadarAgent (fundamental) and ChartPatternAgent (technical)
   in parallel via asyncio.to_thread — each in its own try/except block.
3. If one agent fails, the other still contributes its signals.
4. Return a consolidated AnalysisResponse with status:
   - 'success' → both agents returned results
   - 'partial'  → at least one agent succeeded
   - 'error'    → market data fetch itself failed (nothing to analyse)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from agents.chart_intelligence import ChartPatternAgent, PatternResult
from agents.opportunity_radar import FilingSignal, OpportunityRadarAgent
from models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    DataPreview,
    FilingSignalSchema,
    PatternResultSchema,
)
from services.market_data import MarketDataService

logger = logging.getLogger(__name__)

# ─── Router Instance ──────────────────────────────────────────────────────────
api_v1_router = APIRouter(tags=["Analysis"])


# ─── Dependencies ─────────────────────────────────────────────────────────────

def get_market_data_service() -> MarketDataService:
    """FastAPI dependency: provides a MarketDataService instance."""
    return MarketDataService()


def get_opportunity_radar() -> OpportunityRadarAgent:
    """FastAPI dependency: provides an OpportunityRadarAgent instance."""
    return OpportunityRadarAgent()


def get_chart_agent() -> ChartPatternAgent:
    """FastAPI dependency: provides a ChartPatternAgent instance."""
    return ChartPatternAgent()


MarketDataServiceDep = Annotated[MarketDataService, Depends(get_market_data_service)]
OpportunityRadarDep = Annotated[OpportunityRadarAgent, Depends(get_opportunity_radar)]
ChartAgentDep = Annotated[ChartPatternAgent, Depends(get_chart_agent)]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _signal_to_schema(sig: FilingSignal) -> FilingSignalSchema:
    return FilingSignalSchema(
        ticker=sig.ticker,
        signal_type=sig.signal_type,
        confidence=sig.confidence,
        summary=sig.summary,
        metadata=sig.metadata,
    )


def _pattern_to_schema(pat: PatternResult) -> PatternResultSchema:
    return PatternResultSchema(
        pattern_name=pat.pattern_name,
        direction=pat.direction,
        confidence=pat.confidence,
        start_index=pat.start_index,
        end_index=pat.end_index,
        narrative=pat.narrative,
        metadata=pat.metadata,
    )


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@api_v1_router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Full NSE Stock Analysis",
    response_description="Consolidated fundamental + technical AI-generated insights.",
    status_code=status.HTTP_200_OK,
)
async def analyze_stock(
    request: AnalysisRequest,
    svc: MarketDataServiceDep,
    radar: OpportunityRadarDep,
    chart: ChartAgentDep,
) -> AnalysisResponse:
    """
    Run a full fundamental + technical analysis for an NSE-listed equity.

    **Step 1** — Fetch historical OHLCV data and company news in parallel.

    **Step 2** — Run both AI agents concurrently:
    - `OpportunityRadarAgent` analyses news headlines for investment signals.
    - `ChartPatternAgent` runs RSI, SMA, Bollinger Band checks and generates
      a plain-English narrative via Groq (llama-3.3-70b-versatile).

    **Step 3** — Aggregate results. If one agent fails, the other still
    contributes; the response `status` is set to `'partial'` in that case.

    **Request Body**

    | Field    | Type | Default | Description                        |
    |----------|------|---------|-------------------------------------|
    | ticker   | str  | —       | NSE symbol, e.g. `"TCS"`           |
    | period   | str  | `1mo`   | Look-back window (yfinance string)  |
    | interval | str  | `1d`    | Candle granularity                 |

    **Response Status Values**

    | Status    | Meaning                                       |
    |-----------|-----------------------------------------------|
    | `success` | Both agents returned results                  |
    | `partial` | One agent failed; other succeeded             |
    | `error`   | Market data fetch failed; no analysis done    |
    """
    logger.info(
        "POST /analyze | ticker=%s period=%s interval=%s",
        request.ticker, request.period, request.interval,
    )

    # ── Step 1: Fetch market data + news in parallel ──────────────────────────
    try:
        market_data, news_data = await asyncio.gather(
            asyncio.to_thread(
                svc.get_nse_data, request.ticker, request.period, request.interval
            ),
            asyncio.to_thread(
                svc.get_company_news, request.ticker, max_items=10
            ),
            return_exceptions=False,
        )
    except ValueError as exc:
        logger.warning("Invalid ticker | %s | %s", request.ticker, exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Data fetch failure | %s | %s", request.ticker, exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"Market data unavailable: {exc}") from exc
    except Exception as exc:
        logger.exception("Unexpected fetch error | %s", request.ticker)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Unexpected error fetching market data.") from exc

    # Build the OHLCV DataFrame for the chart agent
    ohlcv_df = pd.DataFrame(market_data["data"])

    # Attach ticker + articles to news_data for the radar agent
    news_data["ticker"] = request.ticker

    # ── Step 2: Run both agents in parallel, each fault-isolated ─────────────

    fundamental_signals: List[FilingSignalSchema] = []
    technical_signals: List[PatternResultSchema] = []
    fundamental_error: Optional[str] = None
    technical_error: Optional[str] = None

    async def run_radar() -> List[FilingSignalSchema]:
        sigs: List[FilingSignal] = await asyncio.to_thread(
            radar.analyze_corporate_filings, news_data
        )
        return [_signal_to_schema(s) for s in sigs]

    async def run_chart() -> List[PatternResultSchema]:
        pats: List[PatternResult] = await asyncio.to_thread(
            chart.detect_patterns,
            ohlcv_df,
            market_data["ticker"],
            request.period,
        )
        return [_pattern_to_schema(p) for p in pats]

    # Gather with return_exceptions=True so one failure doesn't kill the other
    radar_result, chart_result = await asyncio.gather(
        run_radar(), run_chart(), return_exceptions=True
    )

    if isinstance(radar_result, Exception):
        fundamental_error = str(radar_result)
        logger.error("OpportunityRadar failed | %s | %s", request.ticker, radar_result)
    else:
        fundamental_signals = radar_result  # type: ignore[assignment]

    if isinstance(chart_result, Exception):
        technical_error = str(chart_result)
        logger.error("ChartPatternAgent failed | %s | %s", request.ticker, chart_result)
    else:
        technical_signals = chart_result  # type: ignore[assignment]

    # ── Step 3: Determine status and build response ───────────────────────────
    both_failed = bool(fundamental_error and technical_error)
    any_failed = bool(fundamental_error or technical_error)

    response_status = "success" if not any_failed else ("error" if both_failed else "partial")

    messages = [
        f"Analysed {market_data['ticker']} | "
        f"{market_data['row_count']} candles | "
        f"{len(fundamental_signals)} fundamental signal(s) | "
        f"{len(technical_signals)} technical signal(s)."
    ]
    if fundamental_error:
        messages.append(f"Fundamental agent error: {fundamental_error}")
    if technical_error:
        messages.append(f"Technical agent error: {technical_error}")

    data_preview = DataPreview(
        ticker=market_data["ticker"],
        period=market_data["period"],
        interval=market_data["interval"],
        total_rows=market_data["row_count"],
        preview_rows=market_data["data"][:5],
        columns=market_data["columns"],
    )

    logger.info(
        "/analyze complete | ticker=%s status=%s fundamental=%d technical=%d",
        market_data["ticker"], response_status,
        len(fundamental_signals), len(technical_signals),
    )

    return AnalysisResponse(
        status=response_status,
        message=" | ".join(messages),
        ticker=market_data["ticker"],
        data_preview=data_preview,
        fundamental_signals=fundamental_signals,
        technical_signals=technical_signals,
        fundamental_error=fundamental_error,
        technical_error=technical_error,
    )
