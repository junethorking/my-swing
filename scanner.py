
import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ----------------------------
# Indicators
# ----------------------------

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100)


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def local_lows(series: pd.Series, order: int = 3) -> List[int]:
    vals = series.values
    out = []
    for i in range(order, len(vals) - order):
        window = vals[i-order:i+order+1]
        if np.isfinite(vals[i]) and vals[i] == np.nanmin(window):
            out.append(i)
    return out


def bullish_divergence(
    close: pd.Series,
    rsi: pd.Series,
    lookback: int = 60,
    pivot_order: int = 3,
    min_gap: int = 5,
) -> Tuple[bool, Optional[Tuple[int, int]], float]:
    """
    Bullish divergence:
      price low #2 < price low #1
      RSI low #2 > RSI low #1
    Uses recent local price lows and compares RSI at the same dates.
    Returns (found, pivot indexes, strength_score_0_to_1)
    """
    c = close.iloc[-lookback:].reset_index(drop=True)
    r = rsi.iloc[-lookback:].reset_index(drop=True)

    pivots = local_lows(c, order=pivot_order)
    if len(pivots) < 2:
        return False, None, 0.0

    # Try the most recent low against earlier lows
    for j in range(len(pivots) - 1, 0, -1):
        p2 = pivots[j]
        for i in range(j - 1, -1, -1):
            p1 = pivots[i]
            if p2 - p1 < min_gap:
                continue

            price1, price2 = c.iloc[p1], c.iloc[p2]
            rsi1, rsi2 = r.iloc[p1], r.iloc[p2]

            if price2 < price1 and rsi2 > rsi1:
                price_drop = max(0.0, (price1 - price2) / max(price1, 1e-9))
                rsi_rise = max(0.0, (rsi2 - rsi1) / 30.0)
                strength = min(1.0, 0.55 * min(price_drop / 0.10, 1.0) + 0.45 * min(rsi_rise, 1.0))
                return True, (p1, p2), strength

    return False, None, 0.0


def recent_macd_cross(macd: pd.Series, signal: pd.Series, days: int = 5) -> Tuple[bool, int]:
    diff = macd - signal
    tail = diff.iloc[-(days+1):]
    for k in range(len(tail)-1, 0, -1):
        if tail.iloc[k-1] <= 0 and tail.iloc[k] > 0:
            bars_ago = len(tail) - 1 - k
            return True, bars_ago
    return False, 999


def macd_near_cross(macd: pd.Series, signal: pd.Series, tolerance: float = 0.08) -> bool:
    m = float(macd.iloc[-1])
    s = float(signal.iloc[-1])
    scale = max(abs(m), abs(s), 0.01)
    # MACD still below signal, but close enough to be "approaching"
    return (m <= s) and ((s - m) / scale <= tolerance)


def safe_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default


# ----------------------------
# Universe
# ----------------------------

HEADERS = {"User-Agent": "Mozilla/5.0"}

def fetch_sp500() -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    syms = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    return sorted(set(syms))


def fetch_nasdaq100() -> List[str]:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if "ticker" in cols or "symbol" in cols:
            col = t.columns[cols.index("ticker")] if "ticker" in cols else t.columns[cols.index("symbol")]
            vals = t[col].astype(str).str.replace(".", "-", regex=False).tolist()
            vals = [v for v in vals if 1 <= len(v) <= 8 and v.upper() == v]
            if len(vals) >= 50:
                return sorted(set(vals))
    raise RuntimeError("Could not find Nasdaq-100 ticker table.")


def load_universe(kind: str, custom_file: Optional[str] = None) -> List[str]:
    kind = kind.lower()
    if kind == "sp500":
        return fetch_sp500()
    if kind == "nasdaq100":
        return fetch_nasdaq100()
    if kind == "both":
        return sorted(set(fetch_sp500() + fetch_nasdaq100()))
    if kind == "custom":
        if not custom_file:
            raise ValueError("--custom-file is required for custom universe.")
        p = Path(custom_file)
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
            col = "ticker" if "ticker" in df.columns else df.columns[0]
            return sorted(set(df[col].astype(str).str.upper().str.strip()))
        else:
            return sorted(set(x.strip().upper() for x in p.read_text().splitlines() if x.strip()))
    raise ValueError(f"Unknown universe: {kind}")


# ----------------------------
# Scanner
# ----------------------------

@dataclass
class Settings:
    period: str = "1y"
    interval: str = "1d"
    rsi_max: float = 35.0
    divergence_lookback: int = 60
    macd_cross_days: int = 5
    min_avg_dollar_vol: float = 5_000_000
    min_price: float = 3.0
    max_price: float = 10000.0
    min_score: float = 50.0


def score_candidate(
    rsi_now: float,
    divergence: bool,
    div_strength: float,
    macd_cross: bool,
    macd_near: bool,
    vol_ratio: float,
    above_20dma: bool,
) -> float:
    score = 0.0

    # RSI: max 25
    if rsi_now <= 30:
        score += 25
    elif rsi_now <= 35:
        score += 18
    elif rsi_now <= 40:
        score += 8

    # Divergence: max 30
    if divergence:
        score += 20 + 10 * div_strength

    # MACD: max 25
    if macd_cross:
        score += 25
    elif macd_near:
        score += 15

    # Volume: max 10
    if vol_ratio >= 2.0:
        score += 10
    elif vol_ratio >= 1.5:
        score += 7
    elif vol_ratio >= 1.2:
        score += 4

    # Trend reclaim: max 10
    if above_20dma:
        score += 10

    return round(min(100.0, score), 1)


def scan_one(ticker: str, s: Settings) -> Optional[dict]:
    try:
        df = yf.download(
            ticker,
            period=s.period,
            interval=s.interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is None or len(df) < 80:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                return None

        df = df.dropna(subset=["Close", "Volume"]).copy()
        if len(df) < 80:
            return None

        close = df["Close"].astype(float)
        volume = df["Volume"].astype(float)

        rsi = calc_rsi(close)
        macd, sig, hist = calc_macd(close)

        price = float(close.iloc[-1])
        if not (s.min_price <= price <= s.max_price):
            return None

        avg_dollar_vol = float((close * volume).rolling(20).mean().iloc[-1])
        if not np.isfinite(avg_dollar_vol) or avg_dollar_vol < s.min_avg_dollar_vol:
            return None

        rsi_now = float(rsi.iloc[-1])
        divergence, pivots, div_strength = bullish_divergence(
            close, rsi, lookback=s.divergence_lookback
        )
        cross, bars_ago = recent_macd_cross(macd, sig, days=s.macd_cross_days)
        near = macd_near_cross(macd, sig)

        vol20 = float(volume.rolling(20).mean().iloc[-1])
        vol_ratio = float(volume.iloc[-1] / vol20) if vol20 > 0 else 0.0

        ma20 = float(close.rolling(20).mean().iloc[-1])
        above_20dma = price > ma20

        score = score_candidate(
            rsi_now=rsi_now,
            divergence=divergence,
            div_strength=div_strength,
            macd_cross=cross,
            macd_near=near,
            vol_ratio=vol_ratio,
            above_20dma=above_20dma,
        )

        # Core filter:
        # oversold-ish + either divergence or MACD turn
        qualifies = (
            rsi_now <= s.rsi_max
            and (divergence or cross or near)
            and score >= s.min_score
        )
        if not qualifies:
            return None

        return {
            "Ticker": ticker,
            "Score": score,
            "Price": round(price, 2),
            "RSI14": round(rsi_now, 1),
            "BullDiv": "YES" if divergence else "",
            "MACD_Cross": "YES" if cross else "",
            "Cross_Bars_Ago": bars_ago if cross else "",
            "MACD_Near": "YES" if near else "",
            "Vol_vs_20D": round(vol_ratio, 2),
            "Above_20DMA": "YES" if above_20dma else "",
            "Avg_$Vol_20D": int(avg_dollar_vol),
        }
    except Exception:
        return None


def run_scan(tickers: List[str], s: Settings, pause: float = 0.0) -> pd.DataFrame:
    rows = []
    n = len(tickers)
    for idx, ticker in enumerate(tickers, 1):
        row = scan_one(ticker, s)
        if row:
            rows.append(row)
        if pause > 0:
            time.sleep(pause)
        if idx % 25 == 0 or idx == n:
            print(f"Scanned {idx}/{n} | candidates: {len(rows)}")

    if not rows:
        return pd.DataFrame(columns=[
            "Ticker","Score","Price","RSI14","BullDiv","MACD_Cross",
            "Cross_Bars_Ago","MACD_Near","Vol_vs_20D","Above_20DMA","Avg_$Vol_20D"
        ])
    return pd.DataFrame(rows).sort_values(["Score", "RSI14"], ascending=[False, True])


def main():
    parser = argparse.ArgumentParser(description="RSI + MACD + Bullish Divergence swing scanner")
    parser.add_argument("--universe", choices=["sp500", "nasdaq100", "both", "custom"], default="both")
    parser.add_argument("--custom-file", default=None)
    parser.add_argument("--rsi-max", type=float, default=35)
    parser.add_argument("--min-score", type=float, default=50)
    parser.add_argument("--min-dollar-vol", type=float, default=5_000_000)
    parser.add_argument("--output", default="swing_candidates.csv")
    parser.add_argument("--pause", type=float, default=0.0)
    args = parser.parse_args()

    settings = Settings(
        rsi_max=args.rsi_max,
        min_score=args.min_score,
        min_avg_dollar_vol=args.min_dollar_vol,
    )

    tickers = load_universe(args.universe, args.custom_file)
    print(f"Universe: {args.universe} | {len(tickers)} tickers")

    result = run_scan(tickers, settings, pause=args.pause)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("\n=== TOP CANDIDATES ===")
    if result.empty:
        print("No candidates found.")
    else:
        print(result.head(30).to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    from pathlib import Path
    main()
