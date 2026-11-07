"""동전 던지기 진입 + 변동성 추적손절 추세추종 (Tom Basso / Van Tharp 실험 재현).

선물 대신 자산군별 ETF로 롱·숏을 한다. 종목마다 항상 포지션 하나를 들고 있고,
손절되면 다음날 시가에 새로 동전을 던져 재진입한다.

예) python coinflip.py               # 시드 100개 → results/coinflip.md
    python coinflip.py --seeds 50
"""
from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

import data as data_mod

RESULTS = os.path.join(os.path.dirname(__file__), "results")

# 선물 포트폴리오 대용: 주식지수·채권·금속·에너지·농산물·통화·리츠
ETFS = {
    "SPY": "미국 대형주", "QQQ": "나스닥100", "IWM": "미국 소형주", "EFA": "선진국 주식",
    "EEM": "신흥국 주식", "EWJ": "일본 주식", "FXI": "중국 주식", "EWY": "한국 주식",
    "TLT": "미국 장기채", "IEF": "미국 중기채", "LQD": "회사채",
    "GLD": "금", "SLV": "은", "USO": "원유", "UNG": "천연가스", "DBA": "농산물", "DBC": "원자재 바스켓",
    "UUP": "달러", "FXE": "유로", "FXY": "엔", "VNQ": "미국 리츠", "XLE": "에너지 섹터",
}


@dataclass
class Cfg:
    entry: str = "coin"         # coin: 동전 / trend: 100일 수익률 방향 / trend_long: 추세가 위일 때만 매수 / always_long
    exit: str = "trail"         # trail: 3ATR 추적손절 / symmetric: ±3ATR 고정 손절·익절 (대조군)
    atr_len: int = 20
    stop_atr: float = 3.0
    risk_pct: float = 0.005     # 포지션당 계좌 0.5% 리스크 (22종목 → 최대 11%)
    max_pos_pct: float = 0.25   # 종목당 명목 비중 상한
    max_gross: float = np.inf   # 총노출(명목/자산) 상한. 1.0이면 레버리지 없음
    pyramid: int = 1            # 최대 유닛 수 (1이면 피라미딩 없음)
    pyramid_step_atr: float = 2.0
    fee: float = 0.001          # 편도 수수료
    slippage: float = 0.0005
    borrow_fee: float = 0.005   # 공매도 대차 비용 (연)
    seed: int = 0


def load_etfs(start="2007-01-01", refresh=False):
    os.makedirs(data_mod.CACHE_DIR, exist_ok=True)
    names = list(ETFS)
    missing = [t for t in names if refresh or not os.path.exists(data_mod._cache_path("etf", t))]
    if missing:
        for t, df in data_mod._fetch_us(missing, start).items():
            df.to_csv(data_mod._cache_path("etf", t))
    frames = {}
    for t in names:
        df = pd.read_csv(data_mod._cache_path("etf", t), index_col=0, parse_dates=True)
        frames[t] = data_mod._clean(df, t, 0.6).loc[start:]
    # 모든 ETF가 거래되는 날짜로 맞춤 (UNG·UUP 상장 2007년 이후부터)
    common = sorted(set.intersection(*[set(df.index) for df in frames.values()]))
    return {t: df.loc[common] for t, df in frames.items()}


def simulate(prices: dict[str, pd.DataFrame], cfg: Cfg) -> dict:
    rng = np.random.default_rng(cfg.seed)
    names = list(prices)
    idx = prices[names[0]].index
    O, H, L, C = (np.column_stack([prices[t][k].to_numpy(float) for t in names]) for k in ("Open", "High", "Low", "Close"))
    prev_c = np.vstack([C[:1], C[:-1]])
    tr = np.maximum(H - L, np.maximum(abs(H - prev_c), abs(L - prev_c)))
    atr = pd.DataFrame(tr).rolling(cfg.atr_len).mean().to_numpy()
    mom = pd.DataFrame(C).pct_change(100).to_numpy()
    n_days, n = C.shape

    cash = 1.0
    shares = np.zeros(n)            # +롱 / -숏
    stop = np.full(n, np.nan)
    target = np.full(n, np.nan)
    last_add = np.full(n, np.nan)
    units = np.zeros(n, int)
    pending = np.zeros(n)           # 다음날 시가에 진입할 방향 (+1/-1/0)
    pending_add = np.zeros(n, bool)
    equity = np.empty(n_days)
    gross = np.empty(n_days)
    trades = []                     # (종목, 방향, R)
    entry_px = np.full(n, np.nan)
    entry_risk = np.zeros(n)
    entry_cash = np.zeros(n)
    eq_prev = 1.0
    cost = cfg.fee + cfg.slippage
    start = max(cfg.atr_len, 101)

    def close_pos(i, px, day):
        nonlocal cash
        d = np.sign(shares[i])
        fill = px * (1 - d * cfg.slippage)
        cash += shares[i] * fill - abs(shares[i]) * fill * cfg.fee
        pnl = cash_track[i] + shares[i] * fill - abs(shares[i]) * fill * cfg.fee
        trades.append((names[i], int(d), pnl / entry_risk[i], day - entry_day[i]))
        shares[i] = 0
        units[i] = 0
        pending_add[i] = False
        pending[i] = new_direction(i, day)

    def new_direction(i, day):
        if cfg.entry == "coin":
            return 1.0 if rng.random() < 0.5 else -1.0
        if cfg.entry == "trend":
            return 1.0 if mom[day, i] > 0 else -1.0
        if cfg.entry == "trend_long":
            return 1.0 if mom[day, i] > 0 else 0.0
        return 1.0

    cash_track = np.zeros(n)        # 진입 이후 해당 포지션의 현금 흐름 합계 (거래별 손익 계산용)
    entry_day = np.zeros(n, int)

    for day in range(n_days):
        if day < start:
            equity[day], gross[day] = 1.0, 0.0
            if day == start - 1:
                pending[:] = [new_direction(i, day) for i in range(n)]
            continue
        a_prev = atr[day - 1]
        # 1) 시가 진입 / 피라미딩
        for i in range(n):
            if not np.isfinite(a_prev[i]) or a_prev[i] <= 0:
                continue
            if (pending[i] != 0 and shares[i] == 0) or (pending_add[i] and shares[i] != 0):
                d = pending[i] if shares[i] == 0 else np.sign(shares[i])
                fill = O[day, i] * (1 + d * cfg.slippage)
                rps = cfg.stop_atr * a_prev[i]
                q = min(eq_prev * cfg.risk_pct / rps, eq_prev * cfg.max_pos_pct / fill)
                room = cfg.max_gross * eq_prev - np.abs(shares) @ O[day]
                q = min(q, room / fill)
                if q <= 0:
                    continue
                cash -= d * q * fill + q * fill * cfg.fee
                if shares[i] == 0:
                    cash_track[i] = -(d * q * fill + q * fill * cfg.fee)
                    entry_risk[i] = q * rps
                    entry_day[i] = day
                    stop[i] = fill - d * rps
                    target[i] = fill + d * rps
                else:
                    cash_track[i] -= d * q * fill + q * fill * cfg.fee
                    entry_risk[i] += q * rps
                shares[i] += d * q
                units[i] += 1
                last_add[i] = fill
                pending[i] = 0
                pending_add[i] = False
        # 2) 장중 손절 / (대조군) 익절. 같은 날 둘 다 닿으면 손절
        for i in range(n):
            if shares[i] == 0:
                continue
            d = np.sign(shares[i])
            hit_stop = L[day, i] <= stop[i] if d > 0 else H[day, i] >= stop[i]
            if hit_stop:
                px = min(O[day, i], stop[i]) if d > 0 else max(O[day, i], stop[i])
                close_pos(i, px, day)
                continue
            if cfg.exit == "symmetric":
                hit_tgt = H[day, i] >= target[i] if d > 0 else L[day, i] <= target[i]
                if hit_tgt and day > entry_day[i]:
                    px = max(O[day, i], target[i]) if d > 0 else min(O[day, i], target[i])
                    close_pos(i, px, day)
        # 3) 종가: 추적손절 갱신, 피라미딩 신호, 대차비용, 평가
        for i in range(n):
            if shares[i] == 0:
                if pending[i] == 0 and np.isfinite(mom[day, i]):   # 현금 대기 중이면 매일 다시 판단
                    pending[i] = new_direction(i, day)
                continue
            d = np.sign(shares[i])
            if cfg.exit == "trail":
                trail = C[day, i] - d * cfg.stop_atr * atr[day, i]
                stop[i] = max(stop[i], trail) if d > 0 else min(stop[i], trail)
            if units[i] < cfg.pyramid and d * (C[day, i] - last_add[i]) >= cfg.pyramid_step_atr * atr[day, i]:
                pending_add[i] = True
            if d < 0:
                fee = abs(shares[i]) * C[day, i] * cfg.borrow_fee / 252
                cash -= fee
                cash_track[i] -= fee
        mv = shares @ C[day]
        eq_prev = cash + mv
        equity[day] = eq_prev
        gross[day] = np.abs(shares) @ C[day] / eq_prev if eq_prev > 0 else np.nan

    eq = pd.Series(equity[start:], idx[start:])
    tr_df = pd.DataFrame(trades, columns=["ticker", "dir", "R", "days"])
    return {"equity": eq, "trades": tr_df, "gross": float(np.nanmean(gross[start:]))}


def stats(eq: pd.Series) -> dict:
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    r = eq.pct_change().dropna()
    return {
        "CAGR": (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1,
        "MDD": (eq / eq.cummax() - 1).min(),
        "Sharpe": r.mean() / r.std() * np.sqrt(252),
    }


_PRICES = None


def run_cfg(cfg: Cfg):
    global _PRICES
    if _PRICES is None:                 # 프로세스마다 캐시에서 한 번만 읽음
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _PRICES = load_etfs()
    res = simulate(_PRICES, cfg)
    s = stats(res["equity"])
    t = res["trades"]
    s.update({"거래수": len(t), "승률": (t.R > 0).mean(), "평균R": t.R.mean(),
              "상위5%거래 수익비중": t.R.nlargest(max(1, len(t) // 20)).sum() / t.R[t.R > 0].sum(),
              "평균보유일": t.days.mean(), "평균총노출": res["gross"]})
    return cfg, s, res["equity"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prices = load_etfs(refresh=args.refresh)

    no_cost = dict(fee=0.0, slippage=0.0, borrow_fee=0.0)
    variants = {
        "동전 + 3ATR 추적손절": dict(entry="coin", exit="trail"),
        "동전 + 3ATR 추적손절 (비용 0)": dict(entry="coin", exit="trail", **no_cost),
        "동전 + 5ATR 추적손절": dict(entry="coin", exit="trail", stop_atr=5.0),
        "동전 + 추적손절 + 피라미딩(최대 3유닛)": dict(entry="coin", exit="trail", pyramid=3),
        "동전 + ±3ATR 고정 손절·익절 (대조군, 비용 0)": dict(entry="coin", exit="symmetric", **no_cost),
    }
    fixed = {
        "100일 추세 방향 + 3ATR 추적손절": dict(entry="trend", exit="trail"),
        "100일 추세 방향 + 3ATR 추적손절 (비용 0)": dict(entry="trend", exit="trail", **no_cost),
        "100일 추세 방향 + 5ATR 추적손절": dict(entry="trend", exit="trail", stop_atr=5.0),
        "항상 매수 + 3ATR 추적손절": dict(entry="always_long", exit="trail"),
        "[실행 가능] 추세 위일 때만 매수, 5ATR, 레버리지 없음": dict(entry="trend_long", exit="trail", stop_atr=5.0, max_gross=1.0),
        "[실행 가능] 추세 위일 때만 매수, 5ATR, 리스크 1%, 레버리지 없음": dict(entry="trend_long", exit="trail", stop_atr=5.0, max_gross=1.0, risk_pct=0.01),
    }
    jobs = [Cfg(seed=s, **kw) for kw in variants.values() for s in range(args.seeds)]
    jobs += [Cfg(**kw) for kw in fixed.values()]
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(run_cfg, jobs))

    rows, curves = [], {}
    k = 0
    for name in variants:
        res = out[k:k + args.seeds]
        k += args.seeds
        df = pd.DataFrame([s for _, s, _ in res])
        rows.append((name, df))
        curves[name] = pd.concat([e for _, _, e in res], axis=1)
    for name in fixed:
        _, s, e = out[k]
        k += 1
        rows.append((name, pd.DataFrame([s])))
        curves[name] = e.to_frame()

    first = next(iter(curves.values())).index
    spy = prices["SPY"]["Close"].loc[first[0]:]
    ief = prices["IEF"]["Close"].loc[first[0]:]
    bench = {"SPY 보유": spy / spy.iloc[0],
             "SPY 60 / IEF 40 (매일 리밸런싱)": (1 + 0.6 * spy.pct_change().fillna(0) + 0.4 * ief.pct_change().fillna(0)).cumprod()}

    lines = [f"# 동전 던지기 추세추종 ({len(prices)}개 ETF 롱·숏, {first[0]:%Y-%m-%d} ~ {first[-1]:%Y-%m-%d})\n\n"
             f"동전 전략은 시드 {args.seeds}개의 중앙값과 5~95% 범위. 비용: 편도 수수료 0.1%, 슬리피지 0.05%, 공매도 대차비 연 0.5%. "
             "현금 이자는 반영하지 않음.\n\n"
             "| 전략 | CAGR | MDD | Sharpe | 시드 중 CAGR>0 | 승률 | 평균R | 상위 5% 거래의 수익 비중 | 평균 보유일 | 평균 총노출 |\n"
             "|---|---|---|---|---|---|---|---|---|---|\n"]
    for name, df in rows:
        def cell(col, f):
            if len(df) == 1:
                return f(df[col].iloc[0])
            q = df[col].quantile([0.05, 0.5, 0.95])
            return f"{f(q[0.5])} ({f(q[0.05])}~{f(q[0.95])})"
        pct = lambda v: f"{v:.1%}"
        lines.append(f"| {name} | {cell('CAGR', pct)} | {cell('MDD', pct)} | {cell('Sharpe', lambda v: f'{v:.2f}')} | "
                     f"{(df.CAGR > 0).mean():.0%} | {df['승률'].median():.0%} | {df['평균R'].median():+.2f} | "
                     f"{df['상위5%거래 수익비중'].median():.0%} | {df['평균보유일'].median():.0f} | {df['평균총노출'].median():.0%} |\n")
    for name, eq in bench.items():
        s = stats(eq)
        lines.append(f"| {name} | {s['CAGR']:.1%} | {s['MDD']:.1%} | {s['Sharpe']:.2f} | - | - | - | - | - | 100% |\n")
    lines.append("\n종목: " + ", ".join(f"{t}({v})" for t, v in ETFS.items()) + "\n")

    os.makedirs(RESULTS, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5.5))
        coin = curves["동전 + 3ATR 추적손절"]
        ax.plot(coin.index, coin.values, color="tab:blue", alpha=0.06, lw=0.8)
        ax.plot(coin.index, coin.median(axis=1), color="tab:blue", lw=2, label="coin flip + 3ATR trail, with cost (median of 100 seeds)")
        free = curves["동전 + 3ATR 추적손절 (비용 0)"]
        ax.plot(free.index, free.median(axis=1), color="tab:blue", lw=1.5, ls="--", label="coin flip + 3ATR trail, no cost (median)")
        tr = curves["100일 추세 방향 + 3ATR 추적손절 (비용 0)"].iloc[:, 0]
        ax.plot(tr.index, tr.values, color="tab:green", lw=1.5, ls="--", label="100d trend long/short + 3ATR trail, no cost")
        lo = curves["[실행 가능] 추세 위일 때만 매수, 5ATR, 레버리지 없음"].iloc[:, 0]
        ax.plot(lo.index, lo.values, color="tab:orange", lw=2, label="long-only trend, 5ATR, no leverage (with cost)")
        for (name, eq), col in zip(bench.items(), ["gray", "black"]):
            ax.plot(eq.index, eq.values, color=col, lw=1.2, alpha=0.7, label=["SPY buy&hold", "60/40"][col == "black"])
        ax.set_yscale("log")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title("Coin-flip trend following on 22 ETFs (log)")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "coinflip.png"), dpi=110)
        lines.append("\n![coinflip](coinflip.png)\n")
    except ImportError:
        pass

    with open(os.path.join(RESULTS, "coinflip.md"), "w") as f:
        f.write("".join(lines))
    print("".join(lines))


if __name__ == "__main__":
    main()
