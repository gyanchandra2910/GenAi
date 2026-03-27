"""
services/market_data.py — NSE Market Data Service
===================================================
Fetches real-time quotes, historical OHLCV candles, and company news for
NSE-listed equities using the yfinance library.

All tickers follow the NSE convention: symbol + ".NS" suffix
(e.g. "TCS" → "TCS.NS", "RELIANCE" → "RELIANCE.NS").

Design Notes
------------
- Methods are synchronous (yfinance is blocking I/O under the hood).
  Wrap with ``asyncio.to_thread()`` in async route handlers.
- Returns plain Python dicts / lists so that FastAPI can serialise them
  directly without additional Pandas dependency in the API layer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MarketDataService:
    """
    Service class responsible for all market data retrieval.

    Keeps yfinance interactions isolated from agent/router logic so that
    the data source can be swapped (e.g., to a paid vendor feed) without
    touching any other layer.

    Examples
    --------
    >>> svc = MarketDataService()
    >>> df_dict = svc.get_nse_data("TCS", period="1mo")
    >>> news   = svc.get_company_news("INFY")
    """

    NSE_SUFFIX: str = ".NS"

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _nse_ticker(self, ticker: str) -> str:
        """
        Normalise a plain NSE symbol to its yfinance-compatible form.

        Parameters
        ----------
        ticker : str
            Raw symbol, e.g. ``"TCS"``, ``"RELIANCE"``.  The suffix is
            appended only if not already present.

        Returns
        -------
        str
            Normalised symbol, e.g. ``"TCS.NS"``.
        """
        ticker = ticker.strip().upper()
        if not ticker.endswith(self.NSE_SUFFIX):
            ticker += self.NSE_SUFFIX
        return ticker

    # ── Public Interface ──────────────────────────────────────────────────────

    def get_nse_data(
        self,
        ticker: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> Dict[str, Any]:
        """
        Fetch historical OHLCV candles for an NSE-listed equity.

        Parameters
        ----------
        ticker : str
            NSE trading symbol (without the ".NS" suffix), e.g. ``"TCS"``.
        period : str, optional
            Look-back window supported by yfinance:
            ``"1d"``, ``"5d"``, ``"1mo"``, ``"3mo"``, ``"6mo"``,
            ``"1y"``, ``"2y"``, ``"5y"``, ``"10y"``, ``"ytd"``, ``"max"``.
            Defaults to ``"1mo"``.
        interval : str, optional
            Candle granularity: ``"1m"``, ``"5m"``, ``"15m"``, ``"1h"``,
            ``"1d"``, ``"1wk"``, ``"1mo"``. Defaults to ``"1d"``.

        Returns
        -------
        dict
            Keys:

            - ``"ticker"`` — normalised yfinance symbol
            - ``"period"`` — requested look-back
            - ``"interval"`` — candle granularity
            - ``"row_count"`` — number of rows returned
            - ``"columns"`` — list of column names
            - ``"data"`` — list of row dicts (index converted to ISO strings
              for JSON serialisation)

        Raises
        ------
        ValueError
            If *ticker* is empty or the returned DataFrame is empty
            (invalid symbol or no market data for the requested period).
        RuntimeError
            If yfinance raises an unexpected network or parsing error.

        Example
        -------
        >>> svc = MarketDataService()
        >>> result = svc.get_nse_data("RELIANCE", period="5d")
        >>> print(result["row_count"])
        5
        """
        if not ticker:
            raise ValueError("Ticker symbol must be a non-empty string.")

        symbol = self._nse_ticker(ticker)
        logger.info("Fetching OHLCV | symbol=%s period=%s interval=%s", symbol, period, interval)

        try:
            yf_ticker = yf.Ticker(symbol)
            df: pd.DataFrame = yf_ticker.history(period=period, interval=interval)
        except Exception as exc:
            logger.error("yfinance error for %s: %s", symbol, exc)
            raise RuntimeError(
                f"Failed to fetch market data for '{symbol}' from yfinance: {exc}"
            ) from exc

        if df.empty:
            raise ValueError(
                f"No market data returned for '{symbol}'. "
                "Please verify the ticker is valid and listed on NSE."
            )

        # Drop timezone info from DatetimeIndex and convert to ISO strings
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index = df.index.strftime("%Y-%m-%d %H:%M:%S")

        # Retain only standard OHLCV + extra yfinance columns
        keep_cols = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]
        df = df[[c for c in keep_cols if c in df.columns]]

        # Round float columns to 4 decimal places for cleaner output
        float_cols = df.select_dtypes(include="float").columns
        df[float_cols] = df[float_cols].round(4)

        records: List[Dict[str, Any]] = df.reset_index().rename(
            columns={"index": "Date"}
        ).to_dict(orient="records")

        logger.info("OHLCV fetch complete | symbol=%s rows=%d", symbol, len(records))

        return {
            "ticker": symbol,
            "period": period,
            "interval": interval,
            "row_count": len(records),
            "columns": list(df.columns),
            "data": records,
        }

    def get_company_news(
        self,
        ticker: str,
        max_items: int = 10,
    ) -> Dict[str, Any]:
        """
        Fetch the latest news articles for an NSE-listed company.

        News items are sourced from yfinance's ``Ticker.news`` property
        (Yahoo Finance news feed) and are suitable for feeding the
        :class:`~agents.opportunity_radar.OpportunityRadarAgent`.

        Parameters
        ----------
        ticker : str
            NSE trading symbol (without ".NS" suffix), e.g. ``"INFY"``.
        max_items : int, optional
            Maximum number of news articles to return. Defaults to ``10``.

        Returns
        -------
        dict
            Keys:

            - ``"ticker"`` — normalised yfinance symbol
            - ``"count"`` — number of articles returned
            - ``"articles"`` — list of dicts, each containing:
              ``title``, ``publisher``, ``link``, ``publish_time``
              (ISO string), ``thumbnail`` (str or None).

        Raises
        ------
        ValueError
            If *ticker* is empty.
        RuntimeError
            If the news fetch fails due to a network or API error.

        Example
        -------
        >>> svc = MarketDataService()
        >>> result = svc.get_company_news("TCS", max_items=5)
        >>> print(result["count"])
        5
        """
        if not ticker:
            raise ValueError("Ticker symbol must be a non-empty string.")

        symbol = self._nse_ticker(ticker)
        logger.info("Fetching news | symbol=%s max_items=%d", symbol, max_items)

        try:
            yf_ticker = yf.Ticker(symbol)
            raw_news: List[Dict[str, Any]] = yf_ticker.news or []
        except Exception as exc:
            logger.error("yfinance news error for %s: %s", symbol, exc)
            raise RuntimeError(
                f"Failed to fetch news for '{symbol}': {exc}"
            ) from exc

        articles: List[Dict[str, Any]] = []
        for item in raw_news[:max_items]:
            # yfinance news structure can vary across versions; extract safely
            content: Dict[str, Any] = item.get("content", item)
            pub_time_raw: Optional[int] = content.get("pubDate") or item.get("providerPublishTime")
            publish_time: Optional[str] = None
            if isinstance(pub_time_raw, int):
                publish_time = pd.Timestamp(pub_time_raw, unit="s").isoformat()
            elif isinstance(pub_time_raw, str):
                publish_time = pub_time_raw

            thumbnail_data = content.get("thumbnail") or item.get("thumbnail") or {}
            thumbnail_url: Optional[str] = None
            if isinstance(thumbnail_data, dict):
                resolutions = thumbnail_data.get("resolutions", [])
                thumbnail_url = resolutions[0].get("url") if resolutions else None
            
            title = content.get("title") or item.get("title", "N/A")
            publisher = (
                (content.get("provider") or {}).get("displayName")
                or item.get("publisher", "N/A")
            )
            link = (
                (content.get("canonicalUrl") or {}).get("url")
                or item.get("link", "N/A")
            )

            articles.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "publish_time": publish_time,
                "thumbnail": thumbnail_url,
            })

        logger.info("News fetch complete | symbol=%s articles=%d", symbol, len(articles))

        return {
            "ticker": symbol,
            "count": len(articles),
            "articles": articles,
        }
