"""Collect historical market data from Yahoo Finance.

Downloads daily OHLCV data for indices, commodities, forex, crypto,
and volatility measures. No API key required.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

TICKERS = {
    "indices": {
        "^GSPC": "SP500",
        "^DJI": "DowJones",
        "^IXIC": "NASDAQ",
        "^FTSE": "FTSE100",
        "^GDAXI": "DAX",
        "^N225": "Nikkei225",
        "^HSI": "HangSeng",
        "^RUT": "Russell2000",
    },
    "commodities": {
        "GC=F": "Gold",
        "SI=F": "Silver",
        "CL=F": "CrudeOil_WTI",
        "NG=F": "NatGas",
        "HG=F": "Copper",
    },
    "forex": {
        "EURUSD=X": "EURUSD",
        "GBPUSD=X": "GBPUSD",
        "USDJPY=X": "USDJPY",
        "AUDUSD=X": "AUDUSD",
        "USDCAD=X": "USDCAD",
        "USDCHF=X": "USDCHF",
        "DX-Y.NYB": "DollarIndex",
    },
    "bonds": {
        "^TNX": "Treasury10Y",
        "^FVX": "Treasury5Y",
        "^TYX": "Treasury30Y",
        "^IRX": "Treasury3M",
    },
    "volatility": {
        "^VIX": "VIX",
    },
    "crypto": {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
    },
}

DATA_DIR = Path("data/historical/market")


def download_all(
    start_date: str = "2016-01-01",
    end_date: str = "2026-02-26",
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Download historical prices for all tickers and save to CSV."""
    out = output_dir or DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    all_frames = []

    for category, ticker_map in TICKERS.items():
        symbols = list(ticker_map.keys())
        logger.info(f"Downloading {category}: {symbols}")

        try:
            df = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                group_by="ticker",
                auto_adjust=True,
                threads=True,
            )
        except Exception as e:
            logger.error(f"Failed to download {category}: {e}")
            continue

        if df.empty:
            logger.warning(f"No data for {category}")
            continue

        for yf_symbol, name in ticker_map.items():
            try:
                if len(symbols) == 1:
                    ticker_df = df.copy()
                else:
                    try:
                        ticker_df = df[yf_symbol].copy()
                    except KeyError:
                        logger.warning(f"  {name}: not found in download")
                        continue

                ticker_df = ticker_df.dropna(how="all")
                if ticker_df.empty:
                    continue

                ticker_df = ticker_df.reset_index()

                flat_cols = []
                for c in ticker_df.columns:
                    if isinstance(c, tuple):
                        flat_cols.append(c[0] if c[0] else c[1])
                    else:
                        flat_cols.append(c)
                ticker_df.columns = flat_cols

                # Deduplicate columns
                ticker_df = ticker_df.loc[:, ~ticker_df.columns.duplicated()]

                if "Date" not in ticker_df.columns:
                    for c in ticker_df.columns:
                        if str(c).lower() == "date":
                            ticker_df = ticker_df.rename(columns={c: "Date"})
                            break

                standard_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
                keep_cols = [c for c in standard_cols if c in ticker_df.columns]
                ticker_df = ticker_df[keep_cols].copy()

                ticker_df["Ticker"] = yf_symbol
                ticker_df["Name"] = name
                ticker_df["Category"] = category
                all_frames.append(ticker_df)

                logger.info(f"  {name}: {len(ticker_df)} days")
            except Exception as e:
                logger.warning(f"  Failed to process {name}: {e}")

    if not all_frames:
        logger.error("No data collected")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    csv_path = out / "daily_prices.csv"
    combined.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(combined)} rows to {csv_path}")

    return combined


def get_weekly_summary(
    df: pd.DataFrame, week_start: str, week_end: str
) -> dict:
    """Get market summary for a specific week window."""
    mask = (pd.to_datetime(df["Date"]) >= week_start) & (
        pd.to_datetime(df["Date"]) <= week_end
    )
    week_data = df[mask]

    summary = {}
    for ticker in week_data["Ticker"].unique():
        tdf = week_data[week_data["Ticker"] == ticker].sort_values("Date")
        if len(tdf) < 1:
            continue

        name = tdf["Name"].iloc[0]
        category = tdf["Category"].iloc[0]
        open_price = tdf["Open"].iloc[0] if "Open" in tdf.columns else tdf["Close"].iloc[0]
        close_price = tdf["Close"].iloc[-1]
        high = tdf["High"].max() if "High" in tdf.columns else close_price
        low = tdf["Low"].min() if "Low" in tdf.columns else close_price

        pct_change = ((close_price - open_price) / open_price * 100) if open_price else 0

        summary[name] = {
            "ticker": ticker,
            "category": category,
            "open": round(float(open_price), 2),
            "close": round(float(close_price), 2),
            "high": round(float(high), 2),
            "low": round(float(low), 2),
            "change_pct": round(float(pct_change), 2),
        }

    return summary


def format_market_snapshot(summary: dict) -> str:
    """Format market data into a readable text block for training."""
    lines = []
    for category in ["indices", "bonds", "volatility", "commodities", "forex", "crypto"]:
        cat_items = {k: v for k, v in summary.items() if v["category"] == category}
        if not cat_items:
            continue
        lines.append(f"\n{category.upper()}:")
        for name, data in sorted(cat_items.items()):
            direction = "+" if data["change_pct"] >= 0 else ""
            lines.append(
                f"  {name}: {data['close']} ({direction}{data['change_pct']}%)"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = download_all()
    print(f"\nTotal rows: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Tickers: {df['Ticker'].nunique()}")
