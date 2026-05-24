from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf


TICKER = "MNQ=F"

PACIFIC = ZoneInfo("America/Los_Angeles")

POINT_VALUE = 2.00
COMMISSION_PER_TRADE = 0.25

ON_START = time(15, 0)
ON_END = time(6, 29)
RTH_START = time(6, 30)
FORCED_EXIT = time(13, 25)

BREAK_THRESHOLD = 0.25
RECLAIM_DISTANCE = 1.00

STOP_DISTANCE = 50.00
TRAIL_ACTIVATION_POINTS = 25.00
TRAIL_DISTANCE_POINTS = 12.50

MAX_TRADES_PER_DAY = 3
MAX_TRADES_PER_SIDE = 2

SLIPPAGE_TESTS = [0.25, 0.50, 1.00]


def download_data():
    data = yf.download(
        TICKER,
        period="7d",
        interval="1m",
        progress=False,
        auto_adjust=False,
    )

    if data.empty:
        raise ValueError("No data returned from yfinance.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.dropna()

    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")

    data.index = data.index.tz_convert(PACIFIC)

    return data


def session_dates(data):
    dates = sorted(set(data.index.date))
    return dates


def get_session_data(data, session_date):
    on_start_dt = datetime.combine(
        session_date - timedelta(days=1),
        ON_START,
        tzinfo=PACIFIC,
    )

    on_end_dt = datetime.combine(
        session_date,
        ON_END,
        tzinfo=PACIFIC,
    )

    rth_start_dt = datetime.combine(
        session_date,
        RTH_START,
        tzinfo=PACIFIC,
    )

    forced_exit_dt = datetime.combine(
        session_date,
        FORCED_EXIT,
        tzinfo=PACIFIC,
    )

    overnight = data[
        (data.index >= on_start_dt)
        & (data.index <= on_end_dt)
    ]

    rth = data[
        (data.index >= rth_start_dt)
        & (data.index <= forced_exit_dt)
    ]

    return overnight, rth


def max_streak(values, target):
    streak = 0
    best = 0

    for value in values:
        if value == target:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0

    return best


def calculate_metrics(trades):
    if not trades:
        return {
            "total_trades": 0,
            "net_pnl": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "sharpe_ratio": 0,
            "max_wins_in_row": 0,
            "max_losses_in_row": 0,
            "ambiguous_entry_count": 0,
            "ambiguous_exit_count": 0,
            "gap_through_entry_count": 0,
        }

    df = pd.DataFrame(trades)

    wins = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] <= 0]

    gross_profit = wins["net_pnl"].sum()
    gross_loss = losses["net_pnl"].sum()

    if gross_loss == 0:
        profit_factor = np.inf
    else:
        profit_factor = gross_profit / abs(gross_loss)

    daily_pnl = df.groupby("date")["net_pnl"].sum()

    if len(daily_pnl) > 1 and daily_pnl.std() != 0:
        sharpe_ratio = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    outcomes = ["win" if pnl > 0 else "loss" for pnl in df["net_pnl"]]

    return {
        "total_trades": len(df),
        "net_pnl": round(df["net_pnl"].sum(), 2),
        "win_rate": round((len(wins) / len(df)) * 100, 2),
        "profit_factor": round(profit_factor, 2) if np.isfinite(profit_factor) else "inf",
        "sharpe_ratio": round(sharpe_ratio, 2),
        "max_wins_in_row": max_streak(outcomes, "win"),
        "max_losses_in_row": max_streak(outcomes, "loss"),
        "ambiguous_entry_count": int(df["ambiguous_entry"].sum()),
        "ambiguous_exit_count": int(df["ambiguous_exit"].sum()),
        "gap_through_entry_count": int(df["gap_through_entry"].sum()),
    }


def enter_trade(direction, timestamp, candle, on_high, on_low, slippage, gap_through_entry, ambiguous_entry):
    if direction == "long":
        planned_entry = on_low + RECLAIM_DISTANCE

        if gap_through_entry:
            entry_price = max(float(candle["Open"]), planned_entry) + slippage
        else:
            entry_price = planned_entry + slippage

        stop_price = on_low - STOP_DISTANCE

    else:
        planned_entry = on_high - RECLAIM_DISTANCE

        if gap_through_entry:
            entry_price = min(float(candle["Open"]), planned_entry) - slippage
        else:
            entry_price = planned_entry - slippage

        stop_price = on_high + STOP_DISTANCE

    return {
        "direction": direction,
        "entry_time": timestamp,
        "entry_price": entry_price,
        "planned_entry": planned_entry,
        "stop_price": stop_price,
        "trail_active": False,
        "best_price": entry_price,
        "gap_through_entry": gap_through_entry,
        "ambiguous_entry": ambiguous_entry,
        "ambiguous_exit": False,
    }


def close_trade(trade, exit_time, exit_price, exit_reason, session_date):
    direction = trade["direction"]

    if direction == "long":
        points = exit_price - trade["entry_price"]
    else:
        points = trade["entry_price"] - exit_price

    gross_pnl = points * POINT_VALUE
    net_pnl = gross_pnl - COMMISSION_PER_TRADE

    return {
        "date": str(session_date),
        "direction": direction,
        "entry_time": trade["entry_time"],
        "exit_time": exit_time,
        "entry_price": round(trade["entry_price"], 2),
        "exit_price": round(exit_price, 2),
        "points": round(points, 2),
        "gross_pnl": round(gross_pnl, 2),
        "net_pnl": round(net_pnl, 2),
        "exit_reason": exit_reason,
        "gap_through_entry": trade["gap_through_entry"],
        "ambiguous_entry": trade["ambiguous_entry"],
        "ambiguous_exit": trade["ambiguous_exit"],
    }


def manage_open_trade(trade, timestamp, candle, slippage, session_date):
    high = float(candle["High"])
    low = float(candle["Low"])
    close = float(candle["Close"])

    direction = trade["direction"]

    if direction == "long":
        original_stop_hit = low <= trade["stop_price"]

        if high > trade["best_price"]:
            trade["best_price"] = high

        if trade["best_price"] >= trade["entry_price"] + TRAIL_ACTIVATION_POINTS:
            trade["trail_active"] = True
            new_stop = trade["best_price"] - TRAIL_DISTANCE_POINTS
            trade["stop_price"] = max(trade["stop_price"], new_stop)

        trail_stop_hit = low <= trade["stop_price"]

        if original_stop_hit or trail_stop_hit:
            if high >= trade["entry_price"] + TRAIL_ACTIVATION_POINTS and low <= trade["stop_price"]:
                trade["ambiguous_exit"] = True

            exit_price = trade["stop_price"] - slippage
            reason = "stop_or_trail"
            return close_trade(trade, timestamp, exit_price, reason, session_date), None

    else:
        original_stop_hit = high >= trade["stop_price"]

        if low < trade["best_price"]:
            trade["best_price"] = low

        if trade["best_price"] <= trade["entry_price"] - TRAIL_ACTIVATION_POINTS:
            trade["trail_active"] = True
            new_stop = trade["best_price"] + TRAIL_DISTANCE_POINTS
            trade["stop_price"] = min(trade["stop_price"], new_stop)

        trail_stop_hit = high >= trade["stop_price"]

        if original_stop_hit or trail_stop_hit:
            if low <= trade["entry_price"] - TRAIL_ACTIVATION_POINTS and high >= trade["stop_price"]:
                trade["ambiguous_exit"] = True

            exit_price = trade["stop_price"] + slippage
            reason = "stop_or_trail"
            return close_trade(trade, timestamp, exit_price, reason, session_date), None

    return None, trade


def run_backtest(data, slippage):
    trades = []
    ambiguous_setup_count = 0

    for session_date in session_dates(data):
        overnight, rth = get_session_data(data, session_date)

        if overnight.empty or rth.empty:
            continue

        on_high = float(overnight["High"].max())
        on_low = float(overnight["Low"].min())

        open_trade = None

        trades_today = 0
        longs_today = 0
        shorts_today = 0

        long_armed = False
        short_armed = False

        for timestamp, candle in rth.iterrows():
            high = float(candle["High"])
            low = float(candle["Low"])
            open_price = float(candle["Open"])
            close_price = float(candle["Close"])

            if open_trade is not None:
                closed_trade, open_trade = manage_open_trade(
                    open_trade,
                    timestamp,
                    candle,
                    slippage,
                    session_date,
                )

                if closed_trade is not None:
                    trades.append(closed_trade)

                continue

            if trades_today >= MAX_TRADES_PER_DAY:
                continue

            if high >= on_high + BREAK_THRESHOLD:
                short_armed = True

            if low <= on_low - BREAK_THRESHOLD:
                long_armed = True

            short_entry_price = on_high - RECLAIM_DISTANCE
            long_entry_price = on_low + RECLAIM_DISTANCE

            same_candle_short_ambiguous = (
                high >= on_high + BREAK_THRESHOLD
                and low <= short_entry_price
            )

            same_candle_long_ambiguous = (
                low <= on_low - BREAK_THRESHOLD
                and high >= long_entry_price
            )

            if same_candle_short_ambiguous:
                ambiguous_setup_count += 1

            if same_candle_long_ambiguous:
                ambiguous_setup_count += 1

            can_short = (
                short_armed
                and shorts_today < MAX_TRADES_PER_SIDE
                and low <= short_entry_price
            )

            can_long = (
                long_armed
                and longs_today < MAX_TRADES_PER_SIDE
                and high >= long_entry_price
            )

            if can_short and can_long:
                ambiguous_setup_count += 1
                continue

            if can_short:
                gap_through = open_price < short_entry_price

                open_trade = enter_trade(
                    "short",
                    timestamp,
                    candle,
                    on_high,
                    on_low,
                    slippage,
                    gap_through,
                    same_candle_short_ambiguous,
                )

                trades_today += 1
                shorts_today += 1
                short_armed = False
                continue

            if can_long:
                gap_through = open_price > long_entry_price

                open_trade = enter_trade(
                    "long",
                    timestamp,
                    candle,
                    on_high,
                    on_low,
                    slippage,
                    gap_through,
                    same_candle_long_ambiguous,
                )

                trades_today += 1
                longs_today += 1
                long_armed = False
                continue

        if open_trade is not None:
            final_time = rth.index[-1]
            final_close = float(rth.iloc[-1]["Close"])

            if open_trade["direction"] == "long":
                exit_price = final_close - slippage
            else:
                exit_price = final_close + slippage

            trades.append(
                close_trade(
                    open_trade,
                    final_time,
                    exit_price,
                    "forced_close",
                    session_date,
                )
            )

    return trades, ambiguous_setup_count


def main():
    print("Downloading MNQ 1 minute data...")
    data = download_data()

    all_summaries = []

    for slippage in SLIPPAGE_TESTS:
        print(f"Running backtest with {slippage} points slippage per side...")

        trades, ambiguous_setup_count = run_backtest(data, slippage)
        metrics = calculate_metrics(trades)

        metrics["slippage_points_per_side"] = slippage
        metrics["ambiguous_setup_count"] = ambiguous_setup_count

        all_summaries.append(metrics)

        trades_df = pd.DataFrame(trades)

        if not trades_df.empty:
            trades_df["equity_curve"] = trades_df["net_pnl"].cumsum()
            trades_df.to_csv(f"backtest_trades_slippage_{slippage}.csv", index=False)

            equity_df = trades_df[["date", "exit_time", "equity_curve"]]
            equity_df.to_csv(f"equity_curve_slippage_{slippage}.csv", index=False)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv("backtest_summary.csv", index=False)

    print("\nBacktest summary:")
    print(summary_df.to_string(index=False))

    print("\nFiles created:")
    print("backtest_summary.csv")
    print("backtest_trades_slippage_0.25.csv")
    print("backtest_trades_slippage_0.5.csv")
    print("backtest_trades_slippage_1.0.csv")
    print("equity_curve_slippage_0.25.csv")
    print("equity_curve_slippage_0.5.csv")
    print("equity_curve_slippage_1.0.csv")


if __name__ == "__main__":
    main()