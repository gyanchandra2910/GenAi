"""
agents/chart_intelligence.py — Chart Pattern Recognition Agent
===============================================================
Combines vectorised technical indicator computation (via the `ta` library)
with a Groq LLM (llama-3.3-70b-versatile) narrative generator to produce a
plain-English technical analysis briefing for any NSE equity.

Pipeline
--------
1. Validate and normalise OHLCV DataFrame.
2. Compute indicators: RSI-14, SMA-20, SMA-50, VWAP, Bollinger Bands,
   52-week high/low proximity, and volume spikes.
3. Run rule-based pattern screener (_detect_basic_patterns).
4. Feed top signals into Groq chain for a plain-English narrative.
5. Return PatternResult objects ranked by confidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import settings

logger = logging.getLogger(__name__)


# ─── Data Transfer Object ─────────────────────────────────────────────────────


@dataclass
class PatternResult:
    """
    A detected price pattern or technical signal.

    Attributes
    ----------
    pattern_name : str
        Human-readable label (e.g. ``"RSI Oversold Reversal"``).
    direction : str
        ``"BULLISH"`` | ``"BEARISH"`` | ``"NEUTRAL"``.
    confidence : float
        Detection confidence in [0.0, 1.0].
    start_index : int
        Row index where the pattern begins.
    end_index : int
        Row index where the pattern ends (inclusive).
    narrative : str
        LLM-generated plain-English explanation.
    metadata : dict
        Supporting metrics (RSI value, price levels, etc.).
    """

    pattern_name: str
    direction: str
    confidence: float
    start_index: int
    end_index: int
    narrative: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─── System Prompt ────────────────────────────────────────────────────────────

_CHART_SYSTEM_PROMPT = """\
You are an expert Technical Analyst for the Indian stock market (NSE) with \
deep expertise in price action, momentum indicators, and volume analysis.

You will be given a summary of technical signals detected for a specific stock. \
Your job is to synthesize these signals into a concise, actionable technical \
analysis narrative in plain English.

Your narrative must:
1. Clearly state the current technical setup (bullish/bearish/neutral).
2. Reference specific indicator values provided (RSI, SMA, price levels).
3. Identify the most significant pattern or catalyst.
4. Suggest key price levels to watch (support, resistance).
5. Be written for a retail Indian investor — no jargon without explanation.
6. Be 3–5 sentences. Concise, direct, and actionable.

Do NOT output JSON. Output ONLY the narrative text.
"""

_CHART_USER_PROMPT = """\
Stock: {ticker}
Period Analysed: {period}
Current Price: ₹{current_price}

=== DETECTED TECHNICAL SIGNALS ===
{signals_summary}
===================================

Write a 3–5 sentence plain-English technical analysis narrative for this stock.
"""


# ─── Agent ────────────────────────────────────────────────────────────────────


class ChartPatternAgent:
    """
    Detects technical patterns in NSE OHLCV data and narrates them via Groq LLM.

    The agent uses vectorised `ta` library indicators for pattern detection,
    then calls Groq (llama-3.3-70b-versatile) only for the narrative generation step
    to minimise latency and API cost.

    Parameters
    ----------
    model : str, optional
        Groq model. Defaults to ``settings.GROQ_MODEL``.
    temperature : float, optional
        Sampling temperature (low → deterministic analysis).
    top_n_patterns : int, optional
        Number of top signals to include in the LLM prompt (default 5).
    """

    REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_n_patterns: int = 5,
    ) -> None:
        self.model: str = model or settings.GROQ_MODEL
        self.temperature: float = (
            temperature if temperature is not None else settings.GROQ_TEMPERATURE
        )
        self.top_n_patterns: int = top_n_patterns
        self._chain: Optional[Any] = None
        logger.info(
            "ChartPatternAgent initialised | model=%s top_n=%d",
            self.model,
            self.top_n_patterns,
        )

    # ── Private: LLM Chain ────────────────────────────────────────────────────

    def _get_chain(self) -> Any:
        """Lazily build the ChatPromptTemplate → ChatGroq → StrOutputParser chain."""
        if self._chain is None:
            try:
                from langchain_core.output_parsers import StrOutputParser
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_groq import ChatGroq  # type: ignore

                llm = ChatGroq(
                    model=self.model,
                    temperature=self.temperature,
                    api_key=settings.GROQ_API_KEY,
                )
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", _CHART_SYSTEM_PROMPT),
                        ("human", _CHART_USER_PROMPT),
                    ]
                )
                self._chain = prompt | llm | StrOutputParser()
                logger.debug("ChartPattern LangChain chain compiled.")
            except Exception as exc:
                logger.error("Failed to build Chart chain: %s", exc)
                raise RuntimeError(
                    "Could not build ChartPattern chain — check GROQ_API_KEY."
                ) from exc
        return self._chain

    # ── Private: DataFrame Validation ─────────────────────────────────────────

    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate schema, coerce numeric types, and forward-fill minor gaps.

        Raises
        ------
        ValueError
            If required columns are missing or fewer than 20 rows are present.
        """
        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame missing required columns: {missing}."
            )
        if len(df) < 20:
            raise ValueError(
                f"Insufficient data: need ≥20 rows, got {len(df)}."
            )

        for col in self.REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.ffill(inplace=True)
        return df

    # ── Private: Technical Indicators ─────────────────────────────────────────

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicator columns to the DataFrame using the `ta` library.

        Indicators computed
        -------------------
        - ``rsi_14``     : 14-period RSI (momentum)
        - ``sma_20``     : 20-period Simple Moving Average
        - ``sma_50``     : 50-period Simple Moving Average
        - ``bb_upper``   : Bollinger Band upper (20, 2σ)
        - ``bb_lower``   : Bollinger Band lower (20, 2σ)
        - ``vol_sma_20`` : 20-period volume moving average (for spike detection)

        Parameters
        ----------
        df : pd.DataFrame
            Validated OHLCV DataFrame (lowercase column names).

        Returns
        -------
        pd.DataFrame
            Original DataFrame with additional indicator columns appended.
        """
        try:
            import ta  # type: ignore

            # RSI
            df["rsi_14"] = ta.momentum.RSIIndicator(
                close=df["close"], window=14
            ).rsi()

            # SMAs
            df["sma_20"] = ta.trend.SMAIndicator(
                close=df["close"], window=20
            ).sma_indicator()
            df["sma_50"] = ta.trend.SMAIndicator(
                close=df["close"], window=50
            ).sma_indicator()

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(
                close=df["close"], window=20, window_dev=2
            )
            df["bb_upper"] = bb.bollinger_hband()
            df["bb_lower"] = bb.bollinger_lband()

            # Volume SMA for spike detection
            df["vol_sma_20"] = df["volume"].rolling(window=20).mean()

        except ImportError:
            logger.warning("ta library not available — indicator columns will be NaN.")
            for col in ["rsi_14", "sma_20", "sma_50", "bb_upper", "bb_lower", "vol_sma_20"]:
                df[col] = np.nan

        return df

    # ── Private: Pattern Screener ─────────────────────────────────────────────

    def _detect_basic_patterns(
        self, df: pd.DataFrame
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """
        Apply rule-based heuristics to detect common technical setups.

        Each rule returns a tuple of
        ``(pattern_name, direction, confidence, metadata_dict)``.

        Patterns Detected
        -----------------
        - RSI Oversold / Overbought
        - Golden Cross / Death Cross (SMA20 vs SMA50)
        - Price near 52-week High / Low
        - Bollinger Band Squeeze / Breakout
        - Volume Spike (> 2× 20-day average)

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV + indicator columns.

        Returns
        -------
        list of tuples
            Unsorted list of detected pattern tuples.
        """
        signals: List[Tuple[str, str, float, Dict[str, Any]]] = []
        last = df.iloc[-1]
        n = len(df)

        close = float(last["close"])
        rsi = last.get("rsi_14", np.nan)
        sma20 = last.get("sma_20", np.nan)
        sma50 = last.get("sma_50", np.nan)
        bb_upper = last.get("bb_upper", np.nan)
        bb_lower = last.get("bb_lower", np.nan)
        volume = float(last["volume"])
        vol_sma = last.get("vol_sma_20", np.nan)

        # ── 52-week High / Low ────────────────────────────────────────────────
        lookback = min(252, n)
        high_52w = float(df["high"].iloc[-lookback:].max())
        low_52w = float(df["low"].iloc[-lookback:].min())
        pct_from_high = (close - high_52w) / high_52w * 100
        pct_from_low = (close - low_52w) / low_52w * 100

        if pct_from_high >= -3.0:
            signals.append((
                "Near 52-Week High",
                "BULLISH",
                0.75,
                {"close": close, "high_52w": high_52w, "pct_from_high": round(pct_from_high, 2)},
            ))
        elif pct_from_low <= 5.0:
            signals.append((
                "Near 52-Week Low",
                "BEARISH",
                0.60,
                {"close": close, "low_52w": low_52w, "pct_from_low": round(pct_from_low, 2)},
            ))

        # ── RSI ───────────────────────────────────────────────────────────────
        if not np.isnan(rsi):
            rsi_val = float(rsi)
            if rsi_val < 30:
                signals.append((
                    "RSI Oversold",
                    "BULLISH",
                    0.70,
                    {"rsi_14": round(rsi_val, 2), "threshold": 30},
                ))
            elif rsi_val > 70:
                signals.append((
                    "RSI Overbought",
                    "BEARISH",
                    0.65,
                    {"rsi_14": round(rsi_val, 2), "threshold": 70},
                ))

        # ── SMA Golden/Death Cross ────────────────────────────────────────────
        if not (np.isnan(sma20) or np.isnan(sma50)):
            sma20_v = float(sma20)
            sma50_v = float(sma50)
            # Check if a cross happened within the last 5 candles
            if n >= 6:
                prev = df.iloc[-6]
                prev_sma20 = prev.get("sma_20", np.nan)
                prev_sma50 = prev.get("sma_50", np.nan)
                if not (np.isnan(prev_sma20) or np.isnan(prev_sma50)):
                    if float(prev_sma20) < float(prev_sma50) and sma20_v > sma50_v:
                        signals.append((
                            "Golden Cross (SMA20 > SMA50)",
                            "BULLISH",
                            0.80,
                            {"sma_20": round(sma20_v, 2), "sma_50": round(sma50_v, 2)},
                        ))
                    elif float(prev_sma20) > float(prev_sma50) and sma20_v < sma50_v:
                        signals.append((
                            "Death Cross (SMA20 < SMA50)",
                            "BEARISH",
                            0.80,
                            {"sma_20": round(sma20_v, 2), "sma_50": round(sma50_v, 2)},
                        ))

            # Price vs SMA trend context (always report)
            if close > sma20_v > sma50_v:
                signals.append((
                    "Price Above Both SMAs (Uptrend)",
                    "BULLISH",
                    0.65,
                    {"close": close, "sma_20": round(sma20_v, 2), "sma_50": round(sma50_v, 2)},
                ))
            elif close < sma20_v < sma50_v:
                signals.append((
                    "Price Below Both SMAs (Downtrend)",
                    "BEARISH",
                    0.65,
                    {"close": close, "sma_20": round(sma20_v, 2), "sma_50": round(sma50_v, 2)},
                ))

        # ── Bollinger Band Breakout ───────────────────────────────────────────
        if not (np.isnan(bb_upper) or np.isnan(bb_lower)):
            bb_u = float(bb_upper)
            bb_l = float(bb_lower)
            if close > bb_u:
                signals.append((
                    "Bollinger Band Upper Breakout",
                    "BULLISH",
                    0.70,
                    {"close": close, "bb_upper": round(bb_u, 2)},
                ))
            elif close < bb_l:
                signals.append((
                    "Bollinger Band Lower Breakdown",
                    "BEARISH",
                    0.70,
                    {"close": close, "bb_lower": round(bb_l, 2)},
                ))

        # ── Volume Spike ─────────────────────────────────────────────────────
        if not np.isnan(vol_sma) and float(vol_sma) > 0:
            vol_ratio = volume / float(vol_sma)
            if vol_ratio >= 2.0:
                signals.append((
                    "Unusual Volume Spike",
                    "BULLISH" if close >= float(df.iloc[-2]["close"]) else "BEARISH",
                    min(0.50 + (vol_ratio - 2.0) * 0.1, 0.85),
                    {"volume": int(volume), "avg_volume": int(float(vol_sma)), "ratio": round(vol_ratio, 2)},
                ))

        return signals

    # ── Private: Narrative Generator ─────────────────────────────────────────

    def _generate_narrative(
        self,
        ticker: str,
        period: str,
        current_price: float,
        raw_patterns: List[Tuple[str, str, float, Dict[str, Any]]],
    ) -> str:
        """
        Invoke the Groq LLM to produce a plain-English narrative.

        Parameters
        ----------
        ticker : str
            NSE symbol.
        period : str
            Analysis window (e.g. ``"1mo"``).
        current_price : float
            Latest closing price in INR.
        raw_patterns : list
            Top-N pattern tuples from :meth:`_detect_basic_patterns`.

        Returns
        -------
        str
            LLM-generated narrative string.
        """
        if not raw_patterns:
            return (
                f"{ticker} shows no strong directional signals at the moment. "
                "The stock appears range-bound. Investors should wait for a clear "
                "breakout or breakdown before taking a position."
            )

        # Build signals summary for the prompt
        lines = []
        for name, direction, confidence, meta in raw_patterns[:self.top_n_patterns]:
            meta_str = ", ".join(f"{k}={v}" for k, v in meta.items())
            lines.append(
                f"• {name} [{direction}] (confidence={confidence:.0%}) | {meta_str}"
            )
        signals_summary = "\n".join(lines)

        try:
            chain = self._get_chain()
            narrative: str = chain.invoke(
                {
                    "ticker": ticker,
                    "period": period,
                    "current_price": f"{current_price:,.2f}",
                    "signals_summary": signals_summary,
                }
            )
            return narrative.strip()
        except Exception as exc:
            logger.error("LLM narrative generation failed: %s", exc)
            return (
                f"Technical analysis for {ticker} detected {len(raw_patterns)} signal(s): "
                + "; ".join(p[0] for p in raw_patterns[:3])
                + ". LLM narrative unavailable due to an API error."
            )

    # ── Public Interface ──────────────────────────────────────────────────────

    def detect_patterns(
        self,
        stock_data: pd.DataFrame,
        ticker: str = "UNKNOWN",
        period: str = "1mo",
    ) -> List[PatternResult]:
        """
        Full technical analysis pipeline: indicators → pattern detection → LLM narrative.

        Parameters
        ----------
        stock_data : pd.DataFrame
            OHLCV DataFrame with columns ``open``, ``high``, ``low``,
            ``close``, ``volume`` (case-insensitive). Minimum 20 rows.
        ticker : str, optional
            NSE symbol used in the LLM prompt (e.g. ``"TCS.NS"``).
        period : str, optional
            Look-back window label for the LLM prompt (e.g. ``"1mo"``).

        Returns
        -------
        list[PatternResult]
            Detected patterns sorted by confidence (descending).
            Each result has a populated ``narrative`` from the LLM.

        Raises
        ------
        ValueError
            If DataFrame schema validation fails.
        """
        logger.info(
            "detect_patterns | ticker=%s rows=%d columns=%s",
            ticker, len(stock_data), list(stock_data.columns),
        )

        # Step 1 — Validate & normalise
        df = self._validate_dataframe(stock_data)

        # Step 2 — Compute technical indicators
        df = self._compute_indicators(df)

        # Step 3 — Rule-based pattern detection
        raw_patterns = self._detect_basic_patterns(df)

        # Sort by confidence descending; take top N
        raw_patterns.sort(key=lambda x: x[2], reverse=True)
        top_patterns = raw_patterns[:self.top_n_patterns]

        current_price = float(df.iloc[-1]["close"])
        n = len(df)

        logger.info(
            "Pattern detection complete | ticker=%s detected=%d",
            ticker, len(raw_patterns),
        )

        # Step 4 — Generate unified LLM narrative for all top signals
        narrative = self._generate_narrative(ticker, period, current_price, top_patterns)

        # Step 5 — Build PatternResult objects
        if not top_patterns:
            return [
                PatternResult(
                    pattern_name="No Significant Pattern",
                    direction="NEUTRAL",
                    confidence=0.0,
                    start_index=0,
                    end_index=n - 1,
                    narrative=narrative,
                    metadata={"total_rows": n},
                )
            ]

        results: List[PatternResult] = []
        for i, (name, direction, confidence, meta) in enumerate(top_patterns):
            results.append(
                PatternResult(
                    pattern_name=name,
                    direction=direction,
                    confidence=confidence,
                    start_index=max(0, n - 20),
                    end_index=n - 1,
                    # Only the top pattern carries the full LLM narrative
                    narrative=narrative if i == 0 else "",
                    metadata={**meta, "total_rows": n, "current_price": current_price},
                )
            )

        logger.info(
            "detect_patterns complete | ticker=%s results=%d",
            ticker, len(results),
        )
        return results
