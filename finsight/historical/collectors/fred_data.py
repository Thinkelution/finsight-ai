"""Collect historical economic indicators from FRED (Federal Reserve Economic Data).

Uses the FRED public data download endpoints. No API key required for CSV downloads.
"""

import logging
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/historical/market")

FRED_SERIES = {
    "FEDFUNDS": "FedFundsRate",
    "DGS10": "Treasury10Y_Yield",
    "DGS2": "Treasury2Y_Yield",
    "T10Y2Y": "YieldCurve_10Y_2Y",
    "CPIAUCSL": "CPI",
    "UNRATE": "UnemploymentRate",
    "GDP": "GDP",
    "GDPC1": "RealGDP",
    "DTWEXBGS": "DollarIndex_Broad",
    "VIXCLS": "VIX_FRED",
    "BAMLH0A0HYM2": "HighYield_Spread",
    "M2SL": "M2_MoneySupply",
    "UMCSENT": "ConsumerSentiment",
    "PAYEMS": "NonFarmPayrolls",
    "HOUST": "HousingStarts",
    "RSAFS": "RetailSales",
    "PCE": "PersonalConsumption",
    "DCOILWTICO": "CrudeOil_WTI_FRED",
    "GOLDAMGBD228NLBM": "Gold_FRED",
}

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def download_series(
    series_id: str,
    start_date: str = "1976-01-01",
    end_date: str = "2026-02-26",
) -> pd.DataFrame | None:
    """Download a single FRED series as CSV (no API key required)."""
    params = {
        "id": series_id,
        "cosd": start_date,
        "coed": end_date,
    }

    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(FRED_CSV_URL, params=params)
            resp.raise_for_status()

            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))

            date_col = None
            for c in df.columns:
                if "date" in c.lower():
                    date_col = c
                    break
            if date_col is None:
                date_col = df.columns[0]

            df = df.rename(columns={date_col: "Date"})
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"])

            value_col = [c for c in df.columns if c != "Date"][0]
            df = df.rename(columns={value_col: "Value"})
            df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
            df = df.dropna(subset=["Value"])

            return df

    except Exception as e:
        logger.error(f"Failed to download {series_id}: {e}")
        return None


def download_all(
    start_date: str = "1976-01-01",
    end_date: str = "2026-02-26",
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Download all FRED series and combine into a single DataFrame."""
    out = output_dir or DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    all_frames = []

    for series_id, name in FRED_SERIES.items():
        logger.info(f"Downloading FRED {series_id} ({name})...")
        df = download_series(series_id, start_date, end_date)

        if df is not None and not df.empty:
            df["Series"] = series_id
            df["Name"] = name
            all_frames.append(df)
            logger.info(f"  {name}: {len(df)} observations")
        else:
            logger.warning(f"  {name}: no data")

    if not all_frames:
        logger.error("No FRED data collected")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    csv_path = out / "economic_indicators.csv"
    combined.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(combined)} rows to {csv_path}")

    return combined


def get_snapshot(
    df: pd.DataFrame, target_date: str
) -> dict[str, float]:
    """Get the most recent value for each indicator as of a specific date."""
    target = pd.to_datetime(target_date)
    snapshot = {}

    for series_id in df["Series"].unique():
        series_df = df[df["Series"] == series_id]
        name = series_df["Name"].iloc[0]
        before = series_df[series_df["Date"] <= target].sort_values("Date")
        if not before.empty:
            snapshot[name] = round(float(before["Value"].iloc[-1]), 2)

    return snapshot


def format_economic_snapshot(snapshot: dict) -> str:
    """Format economic indicators into readable text."""
    lines = ["ECONOMIC INDICATORS:"]
    for name, value in sorted(snapshot.items()):
        lines.append(f"  {name}: {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = download_all("2020-01-01", "2024-12-31")
    print(f"\nTotal observations: {len(df)}")
    print(f"Series: {df['Series'].nunique()}")

    snapshot = get_snapshot(df, "2024-01-15")
    print(f"\nSnapshot as of 2024-01-15:")
    print(format_economic_snapshot(snapshot))
