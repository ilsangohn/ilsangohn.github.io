"""시세 데이터 로더 (수정주가 기준, 로컬 CSV 캐시)."""
from __future__ import annotations

import os
import warnings

import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")
COLS = ["Open", "High", "Low", "Close", "Volume"]

# 기본 유니버스: 현재 대형주 위주라 생존자 편향이 있다 (상장폐지 종목 미포함).
US_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "XOM", "JNJ", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP",
    "KO", "COST", "AVGO", "WMT", "BAC", "MCD", "CSCO", "ADBE", "CRM", "TMO",
    "ACN", "LIN", "NFLX", "AMD", "ORCL", "TXN", "QCOM", "INTC", "HON", "IBM",
    "CAT", "GS", "AMGN", "DIS", "NKE", "BA", "GE", "MMM", "LOW", "SBUX",
]
US_INDEX = "^GSPC"

KR_TICKERS = [
    "005930", "000660", "373220", "207940", "005380", "000270", "068270", "005490",
    "035420", "035720", "051910", "006400", "105560", "055550", "012330", "028260",
    "066570", "003670", "096770", "034730", "015760", "032830", "086790", "017670",
    "030200", "003550", "009150", "018260", "010130", "011200", "329180", "009540",
    "012450", "042660", "010950", "251270", "036570", "090430", "097950", "316140",
    "024110", "033780", "047050", "247540", "086520", "196170", "028300", "263750",
]
KR_INDEX = "KS11"


def _cache_path(market: str, ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{market}_{ticker.replace('^', '_')}.csv")


def _clean(df: pd.DataFrame, name: str, max_move: float) -> pd.DataFrame:
    df = df[COLS].dropna()
    df = df[pd.to_datetime(df.index).tz_localize(None) < pd.Timestamp.today().normalize()]   # 장중 미완성 봉 제외
    # 거래정지일(시가·고가·저가 0, 거래량 0)은 체결이 불가능하므로 제외
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1) & (df["High"] >= df["Low"])]
    if "Volume" in df and name not in (US_INDEX, KR_INDEX):
        df = df[df["Volume"] > 0]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    jumps = df["Close"].pct_change().abs() > max_move
    if jumps.any():
        warnings.warn(f"{name}: 하루 {max_move:.0%} 넘는 변동 {jumps.sum()}건 — 수정주가가 아닐 수 있음 "
                      f"({', '.join(f'{d:%Y-%m-%d}' for d in df.index[jumps][:3])})")
    return df


def _fetch_us(tickers: list[str], start: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    raw = yf.download(tickers, start=start, auto_adjust=True, group_by="ticker",
                      progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        if len(df.dropna()):
            out[t] = df
    return out


def _fetch_kr(tickers: list[str], start: str) -> dict[str, pd.DataFrame]:
    import FinanceDataReader as fdr
    out = {}
    for t in tickers:
        try:
            out[t] = fdr.DataReader(t, start)
        except Exception as e:  # 한 종목 실패로 전체를 멈추지 않음
            warnings.warn(f"{t}: {e}")
    return out


def load(market: str, tickers: list[str] | None = None, start: str = "2005-01-01",
         csv_dir: str | None = None, refresh: bool = False):
    """(종목별 OHLCV dict, 지수 OHLCV) 반환.

    csv_dir를 주면 그 폴더의 <ticker>.csv 파일(Date,Open,High,Low,Close,Volume)을 읽는다.
    지수 파일 이름은 ^GSPC.csv / KS11.csv.
    """
    tickers = tickers or (US_TICKERS if market == "us" else KR_TICKERS)
    index = US_INDEX if market == "us" else KR_INDEX
    max_move = 0.35 if market == "kr" else 0.6   # 국내 상하한가 ±30%
    names = tickers + [index]

    frames: dict[str, pd.DataFrame] = {}
    if csv_dir:
        for t in names:
            path = os.path.join(csv_dir, f"{t}.csv")
            if os.path.exists(path):
                frames[t] = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        os.makedirs(CACHE_DIR, exist_ok=True)
        missing = [t for t in names if refresh or not os.path.exists(_cache_path(market, t))]
        if missing:
            fetched = _fetch_us(missing, start) if market == "us" else _fetch_kr(missing, start)
            for t, df in fetched.items():
                df.to_csv(_cache_path(market, t))
        for t in names:
            path = _cache_path(market, t)
            if os.path.exists(path):
                frames[t] = pd.read_csv(path, index_col=0, parse_dates=True)

    if index not in frames:
        raise RuntimeError(f"지수 데이터({index})를 불러오지 못했습니다.")
    index_df = _clean(frames.pop(index), index, 0.5)
    data = {t: _clean(df, t, max_move) for t, df in frames.items()}
    data = {t: df.loc[start:] for t, df in data.items() if len(df)}
    return data, index_df.loc[start:]
