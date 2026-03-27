"""
models/schemas.py — Pydantic Request / Response Schemas
=========================================================
Defines the full API contract for the NSE Intelligence Platform.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ─── Request Schemas ──────────────────────────────────────────────────────────


class AnalysisRequest(BaseModel):
    """Request body for POST /api/v1/analyze."""

    ticker: str = Field(
        ..., min_length=1, max_length=20,
        description="NSE trading symbol, e.g. 'TCS', 'RELIANCE'.",
        examples=["TCS"],
    )
    period: str = Field(
        default="1mo",
        description="Look-back window (yfinance period string).",
        examples=["1mo"],
    )
    interval: str = Field(
        default="1d",
        description="Candle granularity (yfinance interval string).",
        examples=["1d"],
    )

    @field_validator("ticker", mode="before")
    @classmethod
    def normalise_ticker(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("period")
    @classmethod
    def validate_period(cls, value: str) -> str:
        allowed = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
        if value not in allowed:
            raise ValueError(f"Invalid period '{value}'. Allowed: {sorted(allowed)}")
        return value

    @field_validator("interval")
    @classmethod
    def validate_interval(cls, value: str) -> str:
        allowed = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"}
        if value not in allowed:
            raise ValueError(f"Invalid interval '{value}'. Allowed: {sorted(allowed)}")
        return value


# ─── Agent Result Schemas ─────────────────────────────────────────────────────


class FilingSignalSchema(BaseModel):
    """Serialisable form of agents.opportunity_radar.FilingSignal."""

    ticker: str = Field(..., description="NSE trading symbol.")
    signal_type: str = Field(..., description="Opportunity category (e.g. BULK_DEAL).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="LLM confidence score.")
    summary: str = Field(..., description="Plain-English opportunity description.")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PatternResultSchema(BaseModel):
    """Serialisable form of agents.chart_intelligence.PatternResult."""

    pattern_name: str = Field(..., description="Chart pattern / technical signal label.")
    direction: str = Field(..., description="BULLISH | BEARISH | NEUTRAL.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence.")
    start_index: int = Field(..., description="Pattern start row index.")
    end_index: int = Field(..., description="Pattern end row index.")
    narrative: str = Field(default="", description="LLM-generated plain-English narrative.")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─── Data Preview Schema ──────────────────────────────────────────────────────


class OHLCVRow(BaseModel):
    """A single OHLCV candle row."""
    Date: str
    Open: Optional[float] = None
    High: Optional[float] = None
    Low: Optional[float] = None
    Close: Optional[float] = None
    Volume: Optional[float] = None
    model_config = {"extra": "allow"}


class DataPreview(BaseModel):
    """Summarised view of fetched market data."""
    ticker: str
    period: str
    interval: str
    total_rows: int
    preview_rows: List[Dict[str, Any]]
    columns: List[str]


# ─── Consolidated Response Schema ─────────────────────────────────────────────


class AnalysisResponse(BaseModel):
    """
    Consolidated response from POST /api/v1/analyze.

    status : 'success' | 'partial' | 'error'
    - 'success'  → both agents returned results
    - 'partial'  → at least one agent succeeded
    - 'error'    → data fetch failed; no analysis possible
    """

    status: str = Field(..., description="'success', 'partial', or 'error'.")
    message: str = Field(..., description="Human-readable result description.")
    ticker: str = Field(..., description="Normalised NSE ticker symbol.")
    data_preview: Optional[DataPreview] = Field(
        default=None, description="OHLCV data preview (first 5 rows)."
    )
    fundamental_signals: List[FilingSignalSchema] = Field(
        default_factory=list,
        description="Opportunity Radar signals from news / filings.",
    )
    technical_signals: List[PatternResultSchema] = Field(
        default_factory=list,
        description="Chart Pattern signals from OHLCV data.",
    )
    fundamental_error: Optional[str] = Field(
        default=None, description="Error from Opportunity Radar agent (if any)."
    )
    technical_error: Optional[str] = Field(
        default=None, description="Error from Chart Pattern agent (if any)."
    )
    error_detail: Optional[str] = Field(
        default=None, description="Top-level error detail when status='error'."
    )
