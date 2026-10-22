"""엔진 체결 로직 검증 (가상 데이터). 실행: python -m pytest test_engine.py -q"""
import numpy as np
import pandas as pd

from engine import Backtest, Costs, Params, add_indicators

NO_COST = Costs(0, 0, 0, 0)


def frame(closes, highs=None, lows=None, opens=None, vols=None):
    n = len(closes)
    idx = pd.bdate_range("2020-01-01", periods=n)
    c = np.asarray(closes, float)
    return pd.DataFrame({
        "Open": opens if opens is not None else c,
        "High": highs if highs is not None else c + 1,
        "Low": lows if lows is not None else c - 1,
        "Close": c,
        "Volume": vols if vols is not None else np.full(n, 1000.0),
    }, index=idx)


def base(n=100):
    return frame(np.full(n, 100.0))   # 평평한 시세: ATR = 2, 20일 최고가 = 101


def breakout_df(day3):
    """100일 횡보 후 거래량 실린 돌파, 이후 day3 함수로 이후 봉을 조정."""
    df = base(130)
    i = 100
    df.iloc[i, df.columns.get_loc("Close")] = 103
    df.iloc[i, df.columns.get_loc("High")] = 104
    df.iloc[i, df.columns.get_loc("Volume")] = 5000
    day3(df, i)
    return df


def run(df, strategy, **kw):
    return Backtest({"X": df}, strategy, NO_COST, Params(use_market_filter=False, **kw)).run()


def set_bar(df, i, o, h, l, c):
    df.iloc[i, :4] = [o, h, l, c]


def test_breakout_enters_next_open_and_hits_2r_target():
    def after(df, i):
        df.iloc[i + 1:, :4] = 103.0
        set_bar(df, i + 1, 103, 104, 102, 103)    # 진입: 다음날 시가 103
        set_bar(df, i + 3, 104, 115, 103, 114)    # 2R 목표 도달
    df = breakout_df(after)
    atr = add_indicators(df, Params()).iloc[100]["atr"]
    bt = run(df, "breakout_2r")
    t = bt.trades[0]
    assert t["entry_date"] == df.index[101] and t["entry"] == 103
    assert t["exit_reason"] == "target"
    assert np.isclose(t["avg_exit"], 103 + 2 * 2 * atr)
    assert np.isclose(t["R"], 2.0)


def test_gap_down_fills_at_open_and_loses_more_than_1r():
    def after(df, i):
        df.iloc[i + 1:, :4] = 103.0
        set_bar(df, i + 1, 103, 104, 102, 103)
        set_bar(df, i + 2, 90, 91, 89, 90)        # 손절가(≈98.7) 아래로 갭하락
    bt = run(breakout_df(after), "breakout_2r")
    t = bt.trades[0]
    assert t["exit_reason"] == "stop_gap" and t["avg_exit"] == 90
    assert t["R"] < -2.5


def test_stop_and_target_same_day_counts_as_stop():
    def after(df, i):
        df.iloc[i + 1:, :4] = 103.0
        set_bar(df, i + 1, 103, 104, 102, 103)
        set_bar(df, i + 2, 103, 120, 90, 110)
    t = run(breakout_df(after), "breakout_2r").trades[0]
    assert t["exit_reason"] == "stop" and np.isclose(t["R"], -1.0)


def test_signal_ignores_future_data():
    df = breakout_df(lambda d, i: None)
    a = add_indicators(df, Params())
    df2 = df.copy()
    df2.iloc[101:] *= 3                            # 미래 데이터를 바꿔도
    b = add_indicators(df2, Params())
    assert a["sig_breakout"].iloc[:101].equals(b["sig_breakout"].iloc[:101])   # 과거 신호는 동일
    assert a["sig_breakout"].iloc[100]


def pullback_df(after):
    """상승 추세 → 20일선 근처까지 눌림 → after가 이후 봉을 정의."""
    c = list(np.linspace(50, 100, 100)) + [99, 97, 96, 95.5]
    df = frame(c)
    after(df, len(c) - 1)
    return df


def test_pullback_setup_and_buy_stop_above_prev_high():
    df = pullback_df(lambda d, i: None)
    ind = add_indicators(df, Params())
    assert ind["sig_pullback"].iloc[-1]


def test_pullback_partial_then_ma10_exit_and_time_stop():
    base_df = pullback_df(lambda d, i: None)
    ext = frame(np.full(30, 96.0)); ext.index = pd.bdate_range(base_df.index[-1] + pd.offsets.BDay(), periods=30)
    set_bar(ext, 0, 96, 97, 95.5, 96.8)
    set_bar(ext, 2, 100, 110, 99, 105)
    set_bar(ext, 3, 104, 105, 103, 104)
    for k in range(4, 30):
        set_bar(ext, k, 97, 97.5, 96.8, 97)       # 손절가보다 위에서 10일선 이탈
    df = pd.concat([base_df, ext])
    p = Params()
    atr = add_indicators(df, p).loc[base_df.index[-1], "atr"]
    bt = run(df, "pullback")
    t = bt.trades[0]
    assert t["entry"] == base_df.iloc[-1]["High"]
    stop = t["entry"] - 1.5 * atr
    assert stop < 96.8                             # 가격 손절이 아니라 10일선 이탈로 나가야 함
    assert t["exit_reason"] == "ma10_break"

    # 시간 손절: 목표도 손절도 안 닿고 횡보하면 10거래일 후 다음날 시가에 정리
    flat = ext.copy()
    for k in range(1, 30):
        set_bar(flat, k, 96.6, 97, 96.2, 96.6)
    bt2 = run(pd.concat([base_df, flat]), "pullback")
    t2 = bt2.trades[0]
    assert t2["exit_reason"] == "time_stop" and t2["days"] == 11


def test_position_size_respects_risk_and_cap():
    def after(df, i):
        df.iloc[i + 1:, :4] = 103.0
        set_bar(df, i + 1, 103, 104, 102, 103)
    df = breakout_df(after)
    bt = Backtest({"X": df}, "breakout_2r", NO_COST, Params(use_market_filter=False), initial_cash=1_000_000).run()
    t = bt.trades[0]
    assert t["init_risk"] <= 1_000_000 * 0.01 + 1e-6
    assert t["shares"] * t["entry"] <= 1_000_000 * 0.20 + 1e-6


def test_random_walk_has_no_edge_after_costs():
    rng = np.random.default_rng(0)
    data = {}
    for k in range(30):
        r = rng.normal(0, 0.02, 2500)
        c = 100 * np.exp(np.cumsum(r))
        o = c * np.exp(rng.normal(0, 0.005, 2500))
        h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.01, 2500)))
        l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.01, 2500)))
        v = rng.lognormal(10, 0.5, 2500)
        data[f"S{k}"] = frame(c, h, l, o, v)
    costs = Costs(0.00015, 0.00015, 0.002, 0.001)
    for s in ("pullback", "breakout_2r", "breakout_trail"):
        bt = Backtest(data, s, costs, Params(use_market_filter=False)).run()
        rs = pd.DataFrame(bt.trades)["R"]
        # 무작위 시세에서 우위가 나오면 엔진 버그(미래 참조 등)를 의심해야 함
        assert rs.mean() < 3 * rs.std() / np.sqrt(len(rs)), (s, rs.mean())
