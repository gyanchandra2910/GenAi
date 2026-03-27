"""
agents/opportunity_radar.py — Corporate Filings Intelligence Agent
===================================================================
Uses a Groq-backed LangChain chain (llama-3.3-70b-versatile) to surface actionable
opportunities hidden within NSE corporate news, bulk deals, insider trades,
earnings beats, and fundraising events.

Pipeline
--------
1. Format raw news / filing data into a structured prompt.
2. Invoke Groq LLM via a ChatPromptTemplate chain.
3. Parse the JSON-list response with JsonOutputParser into FilingSignal objects.
4. Return signals ranked by confidence (descending).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.config import settings

logger = logging.getLogger(__name__)


# ─── Data Transfer Object ─────────────────────────────────────────────────────


@dataclass
class FilingSignal:
    """
    A single investment opportunity signal extracted from corporate filings.

    Attributes
    ----------
    ticker : str
        NSE trading symbol (e.g. ``"RELIANCE"``).
    signal_type : str
        Opportunity category, e.g. ``"BULK_DEAL"``, ``"INSIDER_BUY"``,
        ``"EARNINGS_BEAT"``, ``"FUNDRAISE"``, ``"BOARD_CHANGE"``.
    confidence : float
        LLM-estimated confidence in [0.0, 1.0].
    summary : str
        Plain-English explanation of why this is an opportunity.
    metadata : dict
        Additional context passthrough (raw input, dates, etc.).
    """

    ticker: str
    signal_type: str
    confidence: float
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─── System Prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a Senior NSE Market Analyst with 20 years of experience identifying \
high-conviction investment opportunities from corporate filings and news.

Your task is to analyze the provided news/filing data for an NSE-listed company \
and extract actionable investment signals.

SIGNAL TYPES you must choose from:
- INSIDER_BUY        : Promoter or director purchasing shares
- BULK_DEAL          : Large block transaction by an institution/FII/DII
- EARNINGS_BEAT      : Quarterly results significantly exceed estimates
- FUNDRAISE          : QIP, rights issue, preferential allotment (growth signal)
- BOARD_CHANGE       : New CEO/CFO appointment that is strategically positive
- PARTNERSHIP        : New high-value contract or strategic partnership
- REGULATORY_WIN     : Regulatory approval (drug, SEBI, RBI, etc.)
- BEARISH_FLAG       : Risk factor (litigation, bad earnings, sell-off)
- NEUTRAL            : No actionable signal

CONFIDENCE SCORING GUIDE:
- 0.8–1.0 : Very strong, multiple corroborating data points
- 0.6–0.79: Moderate confidence, single clear signal
- 0.4–0.59: Weak signal, requires further research
- 0.0–0.39: Noise or ambiguous data

OUTPUT FORMAT — return ONLY a valid JSON array, no markdown, no preamble:
[
  {{
    "ticker": "<NSE_SYMBOL>",
    "signal_type": "<SIGNAL_TYPE>",
    "confidence": <0.0-1.0>,
    "summary": "<2-3 sentence plain-English explanation of the opportunity>"
  }}
]

If there are no actionable signals, return an empty array: []
"""

_USER_PROMPT = """\
Analyze the following market data and news for ticker: {ticker}

=== NEWS & FILINGS DATA ===
{filing_data}
===========================

Extract all actionable investment signals and return them as a JSON array.
"""


# ─── Agent ────────────────────────────────────────────────────────────────────


class OpportunityRadarAgent:
    """
    AI agent that scans NSE corporate news and filings for investment signals.

    Uses a deterministic LangChain chain:
        ChatPromptTemplate → ChatGroq (llama-3.3-70b-versatile) → JsonOutputParser

    Parameters
    ----------
    model : str, optional
        Groq model identifier. Defaults to ``settings.GROQ_MODEL``.
    temperature : float, optional
        Sampling temperature (0 = fully deterministic).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.model: str = model or settings.GROQ_MODEL
        self.temperature: float = (
            temperature if temperature is not None else settings.GROQ_TEMPERATURE
        )
        self._chain: Optional[Any] = None
        logger.info(
            "OpportunityRadarAgent initialised | model=%s temperature=%.2f",
            self.model,
            self.temperature,
        )

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _get_chain(self) -> Any:
        """
        Lazily build and cache the full LangChain LCEL chain.

        Chain:  ChatPromptTemplate | ChatGroq | JsonOutputParser

        Returns
        -------
        Runnable
            A compiled LangChain chain ready to be invoked.
        """
        if self._chain is None:
            try:
                from langchain_groq import ChatGroq  # type: ignore

                llm = ChatGroq(
                    model=self.model,
                    temperature=self.temperature,
                    api_key=settings.GROQ_API_KEY,
                )
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", _SYSTEM_PROMPT),
                        ("human", _USER_PROMPT),
                    ]
                )
                self._chain = prompt | llm | JsonOutputParser()
                logger.debug("OpportunityRadar LangChain chain compiled.")
            except Exception as exc:
                logger.error("Failed to build Opportunity Radar chain: %s", exc)
                raise RuntimeError(
                    "Could not build LangChain chain — check GROQ_API_KEY in .env."
                ) from exc
        return self._chain

    @staticmethod
    def _format_filing_data(data: Dict[str, Any]) -> str:
        """
        Convert the raw filing/news dict into a compact, LLM-friendly string.

        Parameters
        ----------
        data : dict
            Expected keys: ``ticker``, ``articles`` (list), optionally
            ``filing_type``, ``content``.

        Returns
        -------
        str
            Plain-text representation suitable for the prompt.
        """
        lines: List[str] = []

        # News articles (from MarketDataService.get_company_news)
        articles: List[Dict[str, Any]] = data.get("articles", [])
        if articles:
            lines.append("--- Recent News ---")
            for i, art in enumerate(articles[:8], 1):
                title = art.get("title", "N/A")
                publisher = art.get("publisher", "N/A")
                pub_time = art.get("publish_time", "N/A")
                lines.append(f"{i}. [{publisher}] {title} (Published: {pub_time})")

        # Structured filings (bulk deals, insider trades, etc.)
        filing_type = data.get("filing_type")
        if filing_type:
            lines.append(f"\n--- Filing ---\nType: {filing_type}")
            content = data.get("content")
            if content:
                lines.append(f"Details: {json.dumps(content, default=str)}")

        return "\n".join(lines) if lines else "No structured data available."

    # ── Public Interface ──────────────────────────────────────────────────────

    def analyze_corporate_filings(
        self,
        data: Dict[str, Any],
    ) -> List[FilingSignal]:
        """
        Analyse NSE news / filing data via a Groq LLM chain and return signals.

        Parameters
        ----------
        data : dict
            Payload containing ``ticker`` and either ``articles`` (list from
            :meth:`~services.market_data.MarketDataService.get_company_news`)
            or ``filing_type`` / ``content`` fields.

        Returns
        -------
        list[FilingSignal]
            Signals ranked by confidence (highest first).
            Empty list if the LLM finds no actionable opportunities.

        Raises
        ------
        ValueError
            If ``data`` is empty.
        RuntimeError
            If the LLM call or JSON parsing fails after exhausting retries.
        """
        if not data:
            raise ValueError("`data` must be a non-empty dictionary.")

        ticker = data.get("ticker", "UNKNOWN")
        logger.info("analyze_corporate_filings | ticker=%s articles=%d",
                    ticker, len(data.get("articles", [])))

        filing_text = self._format_filing_data(data)

        try:
            chain = self._get_chain()
            raw_output: List[Dict[str, Any]] = chain.invoke(
                {"ticker": ticker, "filing_data": filing_text}
            )
        except Exception as exc:
            logger.error("LLM chain invocation failed | ticker=%s | %s", ticker, exc)
            raise RuntimeError(
                f"OpportunityRadarAgent LLM call failed for '{ticker}': {exc}"
            ) from exc

        # Validate and coerce each signal into a FilingSignal dataclass
        signals: List[FilingSignal] = []
        if isinstance(raw_output, list):
            for item in raw_output:
                if not isinstance(item, dict):
                    continue
                try:
                    signals.append(
                        FilingSignal(
                            ticker=item.get("ticker", ticker),
                            signal_type=item.get("signal_type", "NEUTRAL"),
                            confidence=float(item.get("confidence", 0.0)),
                            summary=item.get("summary", ""),
                            metadata={"raw": item},
                        )
                    )
                except (TypeError, ValueError) as parse_err:
                    logger.warning("Skipping malformed signal item: %s | %s", item, parse_err)

        # Sort by confidence descending
        signals.sort(key=lambda s: s.confidence, reverse=True)

        logger.info(
            "analyze_corporate_filings complete | ticker=%s signals=%d",
            ticker, len(signals),
        )
        return signals
