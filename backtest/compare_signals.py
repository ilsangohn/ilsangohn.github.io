"""돌파 신호 정의별 비교 (청산 규칙은 고정) + 구조 기반 자리(피보나치·스프링·지지저항 전환).

예) python compare_signals.py            # 미국+국내 → results/signals.md
"""
from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

import data as data_mod
from engine import BREAKOUT_SIGNALS, COSTS, STRUCTURE_SIGNALS, Backtest, Params, add_indicators, market_ok, summarize
from run import INITIAL_CASH, MIN_TRADED_VALUE, RESULTS

EXITS = ("breakout_2r", "breakout_trail")
EXIT_LABEL = {"breakout_2r": "2R", "breakout_trail": "추적", "structure": "구조"}


def signal_rate(data, index_df, params, col) -> float:
    """필터까지 통과한 신호 수 / 종목-연."""
    mkt = market_ok(index_df)
    n = days = 0
    for df in data.values():
        d = add_indicators(df, params, index_df["Close"]).iloc[250:]
        ok = mkt.reindex(d.index).fillna(False).astype(bool)
        n += int((d[col] & d["liquid"] & ok).sum())
        days += len(d)
    return n / days * 252


def job(args):
    market, signal, exit_, start = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data, index_df = data_mod.load(market, start=start)
    key = "st_signal" if exit_ == "structure" else "bo_signal"
    params = Params(min_traded_value=MIN_TRADED_VALUE[market], **{key: signal})
    bt = Backtest(data, exit_, COSTS[market], params, index_df, INITIAL_CASH[market]).run()
    s = summarize(bt)
    tr = pd.DataFrame(bt.trades)
    eq = bt.equity.index
    mid = eq[0] + (eq[-1] - eq[0]) / 2
    s["R 전반"] = tr.loc[tr.entry_date < mid, "R"].mean() if len(tr) else float("nan")
    s["R 후반"] = tr.loc[tr.entry_date >= mid, "R"].mean() if len(tr) else float("nan")
    if exit_ in (EXITS[0], "structure"):
        col = "sig_structure" if exit_ == "structure" else "sig_breakout"
        s["신호/종목·연"] = signal_rate(data, index_df, params, col)
    return market, signal, exit_, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["us", "kr", "both"], default="both")
    ap.add_argument("--start", default="2005-01-01")
    args = ap.parse_args()
    markets = ["us", "kr"] if args.market == "both" else [args.market]

    for m in markets:   # 다운로드는 한 번만
        data_mod.load(m, start=args.start)
    jobs = [(m, s, e, args.start) for m in markets for s in BREAKOUT_SIGNALS for e in EXITS]
    jobs += [(m, s, "structure", args.start) for m in markets for s in STRUCTURE_SIGNALS]
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(job, jobs))

    os.makedirs(RESULTS, exist_ok=True)
    out = ["# 돌파 신호 정의 비교\n\n청산: 2R 익절(`2R`) / 10일 최저가 추적손절(`추적`) / 구조 목표가·신호일 저가 손절·20일 시간손절(`구조`). 비용·지수필터·수량 규칙은 run.py와 같음.\n"]
    for m in markets:
        out.append(f"\n## {'미국' if m == 'us' else '국내'}\n\n"
                   "| 신호 | 신호/종목·연 | 청산 | 거래수 | 승률 | 평균R | R 전반 | R 후반 | CAGR | MDD |\n"
                   "|---|---|---|---|---|---|---|---|---|---|\n")
        rate = {s: r["신호/종목·연"] for mm, s, e, r in results if mm == m and "신호/종목·연" in r}
        for mm, s, e, r in results:
            if mm != m:
                continue
            out.append(f"| {s} | {rate[s]:.1f} | {EXIT_LABEL[e]} | {r['거래수']} | "
                       f"{r.get('승률', 0):.0%} | {r.get('평균R(기대값)', 0):+.2f} | {r['R 전반']:+.2f} | "
                       f"{r['R 후반']:+.2f} | {r['CAGR']:.1%} | {r['MDD']:.1%} |\n")
    out.append("\n신호 정의\n\n" + "".join(f"- `{k}`: {v}\n" for k, v in {**BREAKOUT_SIGNALS, **STRUCTURE_SIGNALS}.items()))
    path = os.path.join(RESULTS, "signals.md")
    with open(path, "w") as f:
        f.write("".join(out))
    print("".join(out))


if __name__ == "__main__":
    main()
