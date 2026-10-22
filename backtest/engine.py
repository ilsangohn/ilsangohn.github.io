"""단기매매 규칙 백테스트 엔진.

원칙
- 신호는 t일 종가까지의 데이터로만 계산하고, 주문은 t+1일에 체결한다 (look-ahead 방지).
- 손절/목표가는 장중 체결을 가정하되, 갭이 나면 시가로 체결한다.
- 같은 날 손절과 목표가가 모두 닿으면 손절이 먼저 일어났다고 본다 (보수적 가정).
- 진입 당일 저가가 손절가 이하면 그날 손절된 것으로 본다 (보수적 가정).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 설정

@dataclass
class Costs:
    buy_fee: float          # 매수 수수료율
    sell_fee: float         # 매도 수수료율
    sell_tax: float         # 매도 시 세금(거래세 등)
    slippage: float         # 체결가 불리하게 밀리는 비율(편도)


COSTS = {
    # 국내: 온라인 수수료 0.015%, 2026년 코스피 거래세+농특세 합계 0.20%, 슬리피지 0.1%
    "kr": Costs(buy_fee=0.00015, sell_fee=0.00015, sell_tax=0.0020, slippage=0.001),
    # 미국(국내 증권사 경유): 수수료 0.1%, 매도 SEC fee 등 미미, 슬리피지 0.05%
    # 환전 스프레드와 양도소득세(연 250만원 초과분 22%)는 포함하지 않음
    "us": Costs(buy_fee=0.001, sell_fee=0.001, sell_tax=0.0, slippage=0.0005),
}


@dataclass
class Params:
    # 공통
    risk_pct: float = 0.01          # 거래당 계좌 대비 허용 손실
    max_pos_pct: float = 0.20       # 종목당 최대 비중
    max_total_risk: float = 0.06    # 보유 포지션 총 리스크 합계 상한
    max_positions: int = 10
    use_market_filter: bool = True  # 지수 종가 > 지수 200일선일 때만 진입
    min_traded_value: float = 0.0   # 20일 평균 거래대금 하한 (시장별로 run.py에서 지정)
    atr_len: int = 14

    # 1. 눌림목
    pb_near_ma20_atr: float = 0.5   # 저가가 20일선 + 0.5ATR 이내로 내려오면 "근처"
    pb_below_ma20_atr: float = 1.0  # 종가가 20일선 - 1ATR 아래로 깨지면 제외
    pb_pullback_atr: float = 1.0    # 최근 10일 고가 대비 1ATR 이상 조정
    pb_stop_atr: float = 1.5
    pb_target_r: float = 2.0
    pb_time_stop: int = 10

    # 2. 박스권 돌파
    bo_lookback: int = 20
    bo_vol_mult: float = 1.5
    bo_stop_atr: float = 2.0
    bo_target_r: float = 2.0
    bo_trail_len: int = 10
    bo_signal: str = "base"         # BREAKOUT_SIGNALS 중 하나


STRATEGIES = ("pullback", "breakout_2r", "breakout_trail")

# 돌파 신호 정의 (모두 t일 종가 기준)
BREAKOUT_SIGNALS = {
    "base": "20일 최고가 돌파 + 거래량 1.5배",
    "no_volume": "20일 최고가 돌파 (거래량 조건 없음)",
    "donchian55": "55일 최고가 돌파 (터틀 2번 시스템)",
    "high52w": "52주(250일) 최고가 돌파 + 거래량 1.5배",
    "squeeze": "20일 변동폭이 1년 중 하위 25%로 좁아진 뒤 기본 돌파",
    "strong_close": "기본 돌파 + 종가가 당일 고저폭 상위 20%",
    "rel_strength": "기본 돌파 + 120일 수익률이 지수보다 높음",
}


# ---------------------------------------------------------------- 지표 / 신호

def add_indicators(df: pd.DataFrame, p: Params, index_close: pd.Series | None = None) -> pd.DataFrame:
    df = df.copy()
    c, h, l = df["Close"], df["High"], df["Low"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(p.atr_len).mean()
    df["ma10"] = c.rolling(10).mean()
    df["ma20"] = c.rolling(20).mean()
    df["ma60"] = c.rolling(60).mean()
    df["ma20_up"] = df["ma20"] > df["ma20"].shift(5)
    df["hh10"] = h.rolling(10).max()
    df["hh_prev"] = h.rolling(p.bo_lookback).max().shift(1)       # 오늘 제외 최근 N일 최고가
    df["ll_trail"] = l.rolling(p.bo_trail_len).min()               # 오늘 포함 최근 N일 최저가 → 내일 추적손절선
    df["vol_avg_prev"] = df["Volume"].rolling(20).mean().shift(1)
    df["traded_value"] = (c * df["Volume"]).rolling(20).mean()

    df["sig_pullback"] = (
        (c > df["ma60"])
        & df["ma20_up"]
        & (l <= df["ma20"] + p.pb_near_ma20_atr * df["atr"])
        & (c >= df["ma20"] - p.pb_below_ma20_atr * df["atr"])
        & (df["hh10"] - c >= p.pb_pullback_atr * df["atr"])
    )
    df["sig_breakout"] = breakout_signal(df, p, index_close)
    df["liquid"] = df["traded_value"] >= p.min_traded_value
    return df


def breakout_signal(df: pd.DataFrame, p: Params, index_close: pd.Series | None) -> pd.Series:
    c, h, l = df["Close"], df["High"], df["Low"]
    vol_ok = df["Volume"] >= p.bo_vol_mult * df["vol_avg_prev"]
    base = (c > df["hh_prev"]) & vol_ok
    kind = p.bo_signal
    if kind == "base":
        return base
    if kind == "no_volume":
        return c > df["hh_prev"]
    if kind == "donchian55":
        return c > h.rolling(55).max().shift(1)
    if kind == "high52w":
        return (c > h.rolling(250).max().shift(1)) & vol_ok
    if kind == "squeeze":
        width = ((h.rolling(20).max() - l.rolling(20).min()) / c).shift(1)   # 돌파 전날까지의 박스 폭
        return base & (width <= width.rolling(250).quantile(0.25))
    if kind == "strong_close":
        return base & ((c - l) >= 0.8 * (h - l))
    if kind == "rel_strength":
        if index_close is None:
            raise ValueError("rel_strength 신호에는 지수 데이터가 필요합니다")
        idx = index_close.reindex(df.index).ffill()
        return base & (c / c.shift(120) > idx / idx.shift(120))
    raise ValueError(f"알 수 없는 돌파 신호: {kind}")


def market_ok(index_df: pd.DataFrame | None) -> pd.Series | None:
    if index_df is None:
        return None
    c = index_df["Close"]
    return c > c.rolling(200).mean()


# ---------------------------------------------------------------- 시뮬레이션

@dataclass
class Position:
    ticker: str
    shares: int
    entry_date: pd.Timestamp
    entry_price: float
    stop: float
    risk_per_share: float
    target: float | None
    init_shares: int
    init_risk: float
    cash_out: float = 0.0           # 매수에 쓴 총액(수수료 포함)
    cash_in: float = 0.0            # 매도로 받은 총액(수수료·세금 차감)
    days_held: int = 0
    partial_done: bool = False
    exit_next_open: str | None = None
    last_close: float = 0.0
    exits: list = field(default_factory=list)


class Backtest:
    def __init__(self, data: dict[str, pd.DataFrame], strategy: str, costs: Costs,
                 params: Params | None = None, index_df: pd.DataFrame | None = None,
                 initial_cash: float = 100_000_000):
        assert strategy in STRATEGIES, strategy
        self.p = params or Params()
        self.strategy = strategy
        self.costs = costs
        index_close = index_df["Close"] if index_df is not None else None
        self.data = {t: add_indicators(df, self.p, index_close) for t, df in data.items() if len(df) > 80}
        self.rows = {t: df.to_dict("index") for t, df in self.data.items()}   # 날짜 → 봉 (조회 속도용)
        self.mkt = market_ok(index_df) if self.p.use_market_filter else None
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.trades: list[dict] = []
        self.equity_curve: list[tuple] = []

    # --- 체결 헬퍼
    def _buy_cost(self, price, shares):
        return price * shares * (1 + self.costs.buy_fee)

    def _sell_proceeds(self, price, shares):
        return price * shares * (1 - self.costs.sell_fee - self.costs.sell_tax)

    def _sell(self, pos: Position, shares: int, raw_price: float, date, reason: str):
        price = raw_price * (1 - self.costs.slippage)
        proceeds = self._sell_proceeds(price, shares)
        self.cash += proceeds
        pos.cash_in += proceeds
        pos.shares -= shares
        pos.exits.append((date, price, shares, reason))
        if pos.shares == 0:
            self._close_trade(pos, date, reason)

    def _close_trade(self, pos: Position, date, reason: str):
        pnl = pos.cash_in - pos.cash_out
        avg_exit = sum(px * sh for _, px, sh, _ in pos.exits) / pos.init_shares
        self.trades.append({
            "ticker": pos.ticker, "strategy": self.strategy,
            "entry_date": pos.entry_date, "exit_date": date,
            "entry": pos.entry_price, "avg_exit": avg_exit, "shares": pos.init_shares,
            "init_risk": pos.init_risk, "pnl": pnl, "R": pnl / pos.init_risk,
            "days": pos.days_held, "exit_reason": reason,
        })
        del self.positions[pos.ticker]

    # --- 메인 루프
    def run(self) -> "Backtest":
        dates = sorted(set().union(*[df.index for df in self.data.values()]))
        pending: list[dict] = []   # 다음날 체결할 진입 주문
        equity = self.initial_cash
        for d in dates:
            # 1) 보유 포지션 처리: 시가 청산 → 손절 → 목표가
            for t in list(self.positions):
                bar = self.rows[t].get(d)
                if bar is None:
                    continue
                pos = self.positions[t]
                pos.days_held += 1
                if pos.exit_next_open:
                    self._sell(pos, pos.shares, bar["Open"], d, pos.exit_next_open)
                    continue
                if self._check_stop(pos, bar, d):
                    continue
                self._check_target(pos, bar, d)

            # 2) 신규 진입
            pending.sort(key=lambda o: -o["priority"])
            for order in pending:
                self._try_enter(order, d, equity)
            pending = []

            # 3) 종가 기준 처리: 청산 신호, 추적손절 갱신, 평가
            mv = 0.0
            for t, pos in list(self.positions.items()):
                bar = self.rows[t].get(d)
                if bar is not None:
                    pos.last_close = bar["Close"]
                    self._end_of_day(pos, bar)
                mv += pos.shares * pos.last_close
            equity = self.cash + mv
            self.equity_curve.append((d, equity, mv))

            # 4) 내일 진입할 신호 생성
            if self.mkt is not None and not bool(self.mkt.get(d, False)):
                continue
            col = "sig_pullback" if self.strategy == "pullback" else "sig_breakout"
            for t, rows in self.rows.items():
                bar = rows.get(d)
                if t in self.positions or bar is None:
                    continue
                if not (bar[col] and bar["liquid"]) or not np.isfinite(bar["atr"]):
                    continue
                pending.append({
                    "ticker": t, "atr": bar["atr"],
                    # 눌림목: 전일 고가 돌파 시 매수(스톱 주문) / 돌파: 다음날 시가 매수
                    "type": "stop" if self.strategy == "pullback" else "open",
                    "level": bar["High"],
                    "priority": bar["traded_value"],
                })

        # 마지막 날 남은 포지션은 종가로 정리
        last = dates[-1]
        for t, pos in list(self.positions.items()):
            self._sell(pos, pos.shares, pos.last_close, last, "end_of_data")
        self.equity = pd.DataFrame(self.equity_curve, columns=["date", "equity", "invested"]).set_index("date")
        return self

    def _check_stop(self, pos: Position, bar, d) -> bool:
        if bar["Low"] <= pos.stop:
            fill = min(bar["Open"], pos.stop)    # 갭하락이면 시가에 체결
            self._sell(pos, pos.shares, fill, d, "stop" if fill == pos.stop else "stop_gap")
            return True
        return False

    def _check_target(self, pos: Position, bar, d):
        if pos.target is None or pos.partial_done or bar["High"] < pos.target:
            return
        fill = max(bar["Open"], pos.target)      # 갭상승이면 시가에 체결
        if self.strategy == "pullback":
            half = pos.shares // 2
            pos.partial_done = True
            if half > 0:
                self._sell(pos, half, fill, d, "target_half")
        else:
            self._sell(pos, pos.shares, fill, d, "target")

    def _try_enter(self, o: dict, d, equity: float):
        t = o["ticker"]
        bar = self.rows[t].get(d)
        if t in self.positions or bar is None or len(self.positions) >= self.p.max_positions:
            return
        if o["type"] == "open":
            raw = bar["Open"]
        else:
            if bar["High"] < o["level"]:
                return                            # 전일 고가를 못 넘으면 진입 안 함
            raw = max(bar["Open"], o["level"])
        fill = raw * (1 + self.costs.slippage)

        k = self.p.pb_stop_atr if self.strategy == "pullback" else self.p.bo_stop_atr
        rps = k * o["atr"]
        stop = fill - rps
        if stop <= 0:
            return
        shares = math.floor(equity * self.p.risk_pct / rps)
        shares = min(shares, math.floor(equity * self.p.max_pos_pct / fill))
        open_risk = sum(max(p.shares * (p.last_close - p.stop), 0) for p in self.positions.values())
        room = equity * self.p.max_total_risk - open_risk
        shares = min(shares, math.floor(max(room, 0) / rps))
        shares = min(shares, math.floor(self.cash / (fill * (1 + self.costs.buy_fee))))
        if shares < 1:
            return

        target_r = {"pullback": self.p.pb_target_r, "breakout_2r": self.p.bo_target_r}.get(self.strategy)
        cost = self._buy_cost(fill, shares)
        self.cash -= cost
        pos = Position(
            ticker=t, shares=shares, entry_date=d, entry_price=fill, stop=stop,
            risk_per_share=rps, target=fill + target_r * rps if target_r else None,
            init_shares=shares, init_risk=shares * rps, cash_out=cost, last_close=bar["Close"],
        )
        self.positions[t] = pos
        # 진입 당일 손절가 터치 → 손절 (보수적). 당일 목표가 도달은 인정하지 않음.
        if bar["Low"] <= stop:
            self._sell(pos, shares, stop, d, "stop_same_day")

    def _end_of_day(self, pos: Position, bar):
        if self.strategy == "pullback":
            if pos.partial_done and bar["Close"] < bar["ma10"]:
                pos.exit_next_open = "ma10_break"
            elif not pos.partial_done and pos.days_held >= self.p.pb_time_stop:
                pos.exit_next_open = "time_stop"
        elif self.strategy == "breakout_trail":
            if np.isfinite(bar["ll_trail"]):
                pos.stop = max(pos.stop, bar["ll_trail"])


# ---------------------------------------------------------------- 성과 지표

def summarize(bt: Backtest) -> dict:
    eq = bt.equity["equity"]
    tr = pd.DataFrame(bt.trades)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    ret = eq.pct_change().dropna()
    dd = eq / eq.cummax() - 1
    out = {
        "기간": f"{eq.index[0]:%Y-%m-%d} ~ {eq.index[-1]:%Y-%m-%d}",
        "총수익률": eq.iloc[-1] / eq.iloc[0] - 1,
        "CAGR": (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan,
        "MDD": dd.min(),
        "Sharpe": ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan,
        "평균투자비중": (bt.equity["invested"] / eq).mean(),
        "거래수": len(tr),
    }
    if len(tr):
        wins, losses = tr[tr.R > 0], tr[tr.R <= 0]
        streak = (tr.R <= 0).astype(int)
        out.update({
            "승률": len(wins) / len(tr),
            "평균R(기대값)": tr.R.mean(),
            "평균수익R": wins.R.mean() if len(wins) else np.nan,
            "평균손실R": losses.R.mean() if len(losses) else np.nan,
            "최악R": tr.R.min(),
            "ProfitFactor": wins.pnl.sum() / -losses.pnl.sum() if losses.pnl.sum() < 0 else np.nan,
            "최대연속손실": int(streak.groupby((streak != streak.shift()).cumsum()).sum().max()),
            "평균보유일": tr.days.mean(),
        })
    return out


def benchmark(index_df: pd.DataFrame, start, end) -> dict:
    c = index_df["Close"].loc[start:end]
    years = (c.index[-1] - c.index[0]).days / 365.25
    return {"CAGR": (c.iloc[-1] / c.iloc[0]) ** (1 / years) - 1, "MDD": (c / c.cummax() - 1).min()}
