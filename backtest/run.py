"""백테스트 실행.

예)
  python run.py                       # 미국+국내, 3개 전략
  python run.py --market kr --start 2012-01-01
  python run.py --no-market-filter    # 지수 200일선 필터 끄고 비교
  python run.py --sensitivity         # 주요 파라미터를 조금씩 바꿔 견고성 확인
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import warnings

import pandas as pd

import data as data_mod
from engine import COSTS, STRATEGIES, Backtest, Params, benchmark, summarize

RESULTS = os.path.join(os.path.dirname(__file__), "results")
MIN_TRADED_VALUE = {"us": 20e6, "kr": 3e9}          # 20일 평균 거래대금 하한: $2천만 / 30억원
INITIAL_CASH = {"us": 100_000, "kr": 100_000_000}
PCT = {"총수익률", "CAGR", "MDD", "평균투자비중", "승률"}


def fmt(k, v):
    if isinstance(v, str):
        return v
    if pd.isna(v):
        return "-"
    if k in PCT:
        return f"{v:.1%}"
    if isinstance(v, int):
        return str(v)
    return f"{v:.2f}"


def table(rows: dict[str, dict]) -> str:
    keys = list(dict.fromkeys(k for r in rows.values() for k in r))
    head = "| 지표 | " + " | ".join(rows) + " |\n|---|" + "---|" * len(rows) + "\n"
    return head + "".join(
        f"| {k} | " + " | ".join(fmt(k, r.get(k, float("nan"))) for r in rows.values()) + " |\n"
        for k in keys)


def run_one(data, index_df, market, strategy, params, start=None, end=None):
    if start or end:
        data = {t: df.loc[start:end] for t, df in data.items()}
        index_df = index_df.loc[:end]
    return Backtest(data, strategy, COSTS[market], params, index_df, INITIAL_CASH[market]).run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["us", "kr", "both"], default="both")
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--split", help="이 날짜 전/후로 성과를 나눠 비교 (기본: 데이터 기간 중간)")
    ap.add_argument("--csv-dir", help="직접 받은 CSV 폴더 (종목코드.csv)")
    ap.add_argument("--tickers", help="쉼표로 구분한 종목 목록 (기본: 대형주 유니버스)")
    ap.add_argument("--no-market-filter", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="캐시 무시하고 다시 다운로드")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    markets = ["us", "kr"] if args.market == "both" else [args.market]
    report = ["# 단기매매 규칙 백테스트 결과\n"]

    for market in markets:
        tickers = args.tickers.split(",") if args.tickers else None
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data, index_df = data_mod.load(market, tickers, args.start, args.csv_dir, args.refresh)
        notes = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
        params = Params(use_market_filter=not args.no_market_filter,
                        min_traded_value=MIN_TRADED_VALUE[market])

        report.append(f"\n## {'미국' if market == 'us' else '국내'} ({len(data)}종목)\n")
        rows, curves = {}, {}
        for s in STRATEGIES:
            bt = run_one(data, index_df, market, s, params)
            rows[s] = summarize(bt)
            curves[s] = bt.equity["equity"] / bt.equity["equity"].iloc[0]
            pd.DataFrame(bt.trades).to_csv(os.path.join(RESULTS, f"trades_{market}_{s}.csv"), index=False)
            print(f"[{market}] {s}: 거래 {rows[s]['거래수']}건, CAGR {rows[s]['CAGR']:.1%}, MDD {rows[s]['MDD']:.1%}")
        eq0 = curves[STRATEGIES[0]].index
        b = benchmark(index_df, eq0[0], eq0[-1])
        report.append(table(rows))
        report.append(f"\n지수 단순보유: CAGR {b['CAGR']:.1%}, MDD {b['MDD']:.1%}\n")

        # 기간 분할: 앞 구간에서 좋았던 규칙이 뒤 구간에서도 유지되는지
        split = args.split or f"{eq0[0] + (eq0[-1] - eq0[0]) / 2:%Y-%m-%d}"
        split_rows = {}
        for s in STRATEGIES:
            for label, (a, z) in {"전": (None, split), "후": (split, None)}.items():
                r = summarize(run_one(data, index_df, market, s, params, a, z))
                split_rows[f"{s} {label}"] = {k: r.get(k) for k in ("기간", "CAGR", "MDD", "거래수", "승률", "평균R(기대값)")}
        report.append(f"\n### 기간 분할 ({split} 전/후)\n")
        report.append(table(split_rows))

        if args.sensitivity:
            report.append("\n### 파라미터 민감도 (평균R / CAGR)\n")
            grid = {
                "pullback": [("pb_stop_atr", [1.0, 1.5, 2.0]), ("pb_target_r", [1.5, 2.0, 3.0])],
                "breakout_2r": [("bo_lookback", [15, 20, 40]), ("bo_vol_mult", [1.0, 1.5, 2.0])],
                "breakout_trail": [("bo_lookback", [15, 20, 40]), ("bo_trail_len", [5, 10, 20])],
            }
            for s, specs in grid.items():
                for name, values in specs:
                    cells = []
                    for v in values:
                        r = summarize(run_one(data, index_df, market, s, dataclasses.replace(params, **{name: v})))
                        cells.append(f"{v}: {fmt('', r.get('평균R(기대값)', float('nan')))} / {r['CAGR']:.1%}")
                    report.append(f"- {s} `{name}` → " + ", ".join(cells) + "\n")

        if notes:
            report.append("\n<details><summary>데이터 경고</summary>\n\n" + "\n".join(f"- {n}" for n in notes) + "\n</details>\n")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            for s, c in curves.items():
                ax.plot(c.index, c.values, label=s)
            idx = index_df["Close"].loc[eq0[0]:eq0[-1]]
            ax.plot(idx.index, idx / idx.iloc[0], label="index (buy&hold)", color="gray", alpha=0.6)
            ax.set_yscale("log")
            ax.legend()
            ax.set_title(f"{market.upper()} equity (log)")
            fig.tight_layout()
            fig.savefig(os.path.join(RESULTS, f"equity_{market}.png"), dpi=110)
            report.append(f"\n![equity](equity_{market}.png)\n")
        except ImportError:
            pass

    path = os.path.join(RESULTS, "report.md")
    with open(path, "w") as f:
        f.write("".join(report))
    print(f"\n보고서: {path}")


if __name__ == "__main__":
    main()
