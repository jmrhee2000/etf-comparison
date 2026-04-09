"""트레이딩 시그널 분석 엔진 — 펀드매니저 포트폴리오 변화 기반"""
import os
import re
import logging
from datetime import date, timedelta

import pandas as pd
import numpy as np
import yfinance as yf

from collector import load_all_data
from analyzer import get_portfolio_data, get_dates, track_entries_exits

log = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "prices")
os.makedirs(CACHE_DIR, exist_ok=True)


# ── Ticker Mapping ───────────────────────────────────────────────────────────

def _bloomberg_to_yahoo(ticker: str) -> str | None:
    """Convert Bloomberg-style ticker to Yahoo Finance ticker."""
    if not ticker or not isinstance(ticker, str):
        return None
    ticker = ticker.strip()

    # "NVDA US EQUITY" or "NVDA US Equity" → "NVDA"
    m = re.match(r'^([A-Z0-9]+)\s+US\s+(?:EQUITY|Equity)', ticker)
    if m:
        return m.group(1)

    # Korean stock codes (6 digits) → add .KS suffix
    if re.match(r'^\d{6}$', ticker):
        return ticker + ".KS"

    return None


def build_ticker_map(tf_df: pd.DataFrame, ko_df: pd.DataFrame) -> dict:
    """Build mapping: norm_name → yahoo_ticker from both ETFs.

    Accepts both raw DataFrames (with 'ticker'+'name') and
    processed DataFrames (with 'ticker'+'name'+'norm_name').
    """
    from analyzer import _normalize_name
    ticker_map = {}
    for df in [tf_df, ko_df]:
        if df.empty or "ticker" not in df.columns:
            continue
        cols = ["name", "ticker"]
        sub = df[cols].drop_duplicates()
        for _, row in sub.iterrows():
            name = str(row["name"])
            norm = _normalize_name(name)
            if not norm or norm in ticker_map:
                continue
            yahoo = _bloomberg_to_yahoo(str(row.get("ticker", "")))
            if yahoo:
                ticker_map[norm] = yahoo
    return ticker_map


# ── Price Data ───────────────────────────────────────────────────────────────

def _price_cache_path(ticker: str) -> str:
    safe = ticker.replace("/", "_").replace(".", "_")
    return os.path.join(CACHE_DIR, f"{safe}.csv")


def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily close prices with caching."""
    cache = _price_cache_path(ticker)

    if os.path.exists(cache):
        cached = pd.read_csv(cache, parse_dates=["Date"])
        cached_end = cached["Date"].max()
        if pd.Timestamp(end) <= cached_end + pd.Timedelta(days=1):
            return cached

    try:
        data = yf.download(ticker, start=start, end=end, auto_adjust=True,
                           progress=False, timeout=10)
        if data.empty:
            return pd.DataFrame()
        data = data.reset_index()[["Date", "Close"]]
        # Flatten multi-level columns if needed
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
        data.columns = ["Date", "Close"]
        data.to_csv(cache, index=False)
        return data
    except Exception as e:
        log.warning("Failed to fetch prices for %s: %s", ticker, e)
        return pd.DataFrame()


def fetch_all_prices(ticker_map: dict, start_date: date, end_date: date) -> dict:
    """Fetch prices for all tickers. Returns {norm_name: DataFrame}."""
    start_str = (start_date - timedelta(days=30)).isoformat()
    end_str = (end_date + timedelta(days=30)).isoformat()
    prices = {}
    for norm, yahoo_ticker in ticker_map.items():
        df = fetch_prices(yahoo_ticker, start_str, end_str)
        if not df.empty:
            prices[norm] = df
    return prices


# ── Signal Detection ─────────────────────────────────────────────────────────

def detect_signals(portfolio_df: pd.DataFrame, etf_name: str) -> pd.DataFrame:
    """Detect all trading signals from portfolio changes.

    Signal types:
    - NEW_ENTRY: stock first appears in portfolio
    - EXIT: stock completely removed
    - BIG_INCREASE: weight increased >1.5% in a day with qty increase
    - BIG_DECREASE: weight decreased >1.5% in a day with qty decrease
    - CONVICTION_BUY: weight steadily increasing over 5+ days
    """
    if portfolio_df.empty:
        return pd.DataFrame()

    dates = get_dates(portfolio_df)
    signals = []

    for i in range(1, len(dates)):
        prev_date = dates[i-1]
        curr_date = dates[i]

        prev = portfolio_df[portfolio_df["date"] == prev_date].set_index("norm_name")
        curr = portfolio_df[portfolio_df["date"] == curr_date].set_index("norm_name")

        prev_stocks = set(prev.index)
        curr_stocks = set(curr.index)

        # NEW ENTRY signals
        for norm in (curr_stocks - prev_stocks):
            row = curr.loc[norm]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            signals.append({
                "date": curr_date,
                "etf": etf_name,
                "signal": "NEW_ENTRY",
                "name": row["name"],
                "norm_name": norm,
                "weight": row["weight"],
                "detail": f"신규 편입 비중 {row['weight']:.2f}%",
            })

        # EXIT signals
        for norm in (prev_stocks - curr_stocks):
            row = prev.loc[norm]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            signals.append({
                "date": curr_date,
                "etf": etf_name,
                "signal": "EXIT",
                "name": row["name"],
                "norm_name": norm,
                "weight": row["weight"],
                "detail": f"완전 편출 (직전 비중 {row['weight']:.2f}%)",
            })

        # WEIGHT CHANGE signals
        common = prev_stocks & curr_stocks
        for norm in common:
            p = prev.loc[norm]
            c = curr.loc[norm]
            if isinstance(p, pd.DataFrame):
                p = p.iloc[0]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[0]

            w_change = c["weight"] - p["weight"]
            q_prev = p["quantity"]
            q_curr = c["quantity"]
            q_pct = ((q_curr - q_prev) / q_prev * 100) if q_prev > 0 else 0

            if w_change > 0.5 and q_pct > 3:
                # Classify buy intensity
                intensity_score = w_change * 0.6 + abs(q_pct) * 0.02 + c["weight"] * 0.1
                if w_change >= 3.0 or (w_change >= 2.0 and q_pct >= 30):
                    grade = "STRONG"
                elif w_change >= 1.5 or (w_change >= 1.0 and q_pct >= 15):
                    grade = "MODERATE"
                else:
                    grade = "MILD"
                signals.append({
                    "date": curr_date,
                    "etf": etf_name,
                    "signal": "BIG_INCREASE",
                    "name": c["name"],
                    "norm_name": norm,
                    "weight": c["weight"],
                    "detail": f"비중 +{w_change:.2f}%, 수량 +{q_pct:.1f}%",
                    "intensity": grade,
                    "intensity_score": round(intensity_score, 2),
                    "weight_change": w_change,
                    "qty_change_pct": q_pct,
                })
            elif w_change < -1.5 and q_pct < -3:
                signals.append({
                    "date": curr_date,
                    "etf": etf_name,
                    "signal": "BIG_DECREASE",
                    "name": c["name"],
                    "norm_name": norm,
                    "weight": c["weight"],
                    "detail": f"비중 {w_change:.2f}%, 수량 {q_pct:.1f}%",
                })

    # CONVICTION BUY: weight increased in 4+ of last 5 trading days
    for norm in portfolio_df["norm_name"].unique():
        stock = portfolio_df[portfolio_df["norm_name"] == norm].sort_values("date")
        if len(stock) < 5:
            continue

        stock = stock.reset_index(drop=True)
        for i in range(4, len(stock)):
            window = stock.iloc[i-4:i+1]
            w_diffs = window["weight"].diff().dropna()
            if (w_diffs > 0).sum() >= 4:
                total_increase = window["weight"].iloc[-1] - window["weight"].iloc[0]
                if total_increase > 1.0:
                    sig_date = window["date"].iloc[-1]
                    # Avoid duplicate if already signaled
                    if not any(s["date"] == sig_date and s["norm_name"] == norm
                               and s["signal"] == "CONVICTION_BUY" for s in signals):
                        signals.append({
                            "date": sig_date,
                            "etf": etf_name,
                            "signal": "CONVICTION_BUY",
                            "name": window["name"].iloc[-1],
                            "norm_name": norm,
                            "weight": window["weight"].iloc[-1],
                            "detail": f"5일간 지속 비중 증가 (+{total_increase:.2f}%)",
                        })

    df = pd.DataFrame(signals)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    return df


# ── Signal Backtesting ───────────────────────────────────────────────────────

def backtest_signals(signals_df: pd.DataFrame, prices: dict,
                     forward_days: list[int] = [1, 3, 5, 10, 20]) -> pd.DataFrame:
    """Backtest signals against actual stock prices.

    For each signal, calculate the stock's return over the next N days.
    """
    if signals_df.empty:
        return pd.DataFrame()

    results = []
    for _, sig in signals_df.iterrows():
        norm = sig["norm_name"]
        if norm not in prices:
            continue

        price_df = prices[norm].copy()
        price_df["Date"] = pd.to_datetime(price_df["Date"])
        sig_date = pd.Timestamp(sig["date"])

        # Find the closest trading day on or after signal date
        future_prices = price_df[price_df["Date"] >= sig_date].sort_values("Date")
        if future_prices.empty:
            continue

        entry_price = float(future_prices.iloc[0]["Close"])
        entry_date = future_prices.iloc[0]["Date"]

        row = {
            "date": sig["date"],
            "etf": sig["etf"],
            "signal": sig["signal"],
            "name": sig["name"],
            "norm_name": norm,
            "weight": sig["weight"],
            "detail": sig["detail"],
            "entry_price": entry_price,
        }

        for days in forward_days:
            target_date = entry_date + pd.Timedelta(days=days)
            future = price_df[(price_df["Date"] >= target_date)].sort_values("Date")
            if not future.empty:
                exit_price = float(future.iloc[0]["Close"])
                ret = (exit_price - entry_price) / entry_price * 100
                row[f"return_{days}d"] = ret
            else:
                row[f"return_{days}d"] = None

        results.append(row)

    return pd.DataFrame(results)


def summarize_backtest(bt: pd.DataFrame, group_col: str = "signal",
                       return_col: str = "return_5d") -> pd.DataFrame:
    """Summarize backtest results by signal type."""
    if bt.empty or return_col not in bt.columns:
        return pd.DataFrame()

    valid = bt.dropna(subset=[return_col])
    if valid.empty:
        return pd.DataFrame()

    summary = valid.groupby(group_col).agg(
        count=(return_col, "count"),
        avg_return=(return_col, "mean"),
        median_return=(return_col, "median"),
        win_rate=(return_col, lambda x: (x > 0).mean() * 100),
        avg_win=(return_col, lambda x: x[x > 0].mean() if (x > 0).any() else 0),
        avg_loss=(return_col, lambda x: x[x < 0].mean() if (x < 0).any() else 0),
        best=(return_col, "max"),
        worst=(return_col, "min"),
    ).round(2)

    return summary.sort_values("avg_return", ascending=False)


# ── Consensus & Divergence Signals ───────────────────────────────────────────

def find_consensus_signals(tf_signals: pd.DataFrame,
                           ko_signals: pd.DataFrame,
                           window_days: int = 5) -> pd.DataFrame:
    """Find stocks where both funds made similar moves within a time window."""
    if tf_signals.empty or ko_signals.empty:
        return pd.DataFrame()

    results = []
    for _, tf_sig in tf_signals.iterrows():
        # Find Koact signals for the same stock within the window
        mask = (
            (ko_signals["norm_name"] == tf_sig["norm_name"]) &
            (abs((ko_signals["date"] - tf_sig["date"]).dt.days) <= window_days) &
            (ko_signals["signal"] == tf_sig["signal"])
        )
        matches = ko_signals[mask]
        if not matches.empty:
            ko_sig = matches.iloc[0]
            results.append({
                "name": tf_sig["name"],
                "norm_name": tf_sig["norm_name"],
                "signal": tf_sig["signal"],
                "tf_date": tf_sig["date"],
                "ko_date": ko_sig["date"],
                "tf_weight": tf_sig["weight"],
                "ko_weight": ko_sig["weight"],
                "consensus_strength": "STRONG",
            })

    return pd.DataFrame(results)


def find_divergence_signals(tf_df: pd.DataFrame, ko_df: pd.DataFrame) -> pd.DataFrame:
    """Find stocks where one fund is buying while the other is selling."""
    tf_sigs = detect_signals(tf_df, "Timefolio")
    ko_sigs = detect_signals(ko_df, "Koact")

    if tf_sigs.empty or ko_sigs.empty:
        return pd.DataFrame()

    results = []
    buy_signals = {"NEW_ENTRY", "BIG_INCREASE", "CONVICTION_BUY"}
    sell_signals = {"EXIT", "BIG_DECREASE"}

    for _, sig1 in tf_sigs.iterrows():
        mask = (
            (ko_sigs["norm_name"] == sig1["norm_name"]) &
            (abs((ko_sigs["date"] - sig1["date"]).dt.days) <= 10)
        )
        for _, sig2 in ko_sigs[mask].iterrows():
            is_divergent = (
                (sig1["signal"] in buy_signals and sig2["signal"] in sell_signals) or
                (sig1["signal"] in sell_signals and sig2["signal"] in buy_signals)
            )
            if is_divergent:
                results.append({
                    "name": sig1["name"],
                    "norm_name": sig1["norm_name"],
                    "tf_signal": sig1["signal"],
                    "ko_signal": sig2["signal"],
                    "tf_date": sig1["date"],
                    "ko_date": sig2["date"],
                    "tf_detail": sig1["detail"],
                    "ko_detail": sig2["detail"],
                })

    return pd.DataFrame(results)


# ── Current Actionable Insights ──────────────────────────────────────────────

def get_current_signals(signals_df: pd.DataFrame, lookback_days: int = 7) -> pd.DataFrame:
    """Get signals from the last N days."""
    if signals_df.empty:
        return pd.DataFrame()
    cutoff = pd.Timestamp(date.today()) - pd.Timedelta(days=lookback_days)
    recent = signals_df[signals_df["date"] >= cutoff].copy()
    return recent.sort_values("date", ascending=False)


def generate_trading_notes(bt: pd.DataFrame, signals_df: pd.DataFrame) -> list[dict]:
    """Generate human-readable trading insights."""
    notes = []

    if bt.empty:
        return notes

    # 1) Best performing signal types
    for ret_col in ["return_5d", "return_10d", "return_20d"]:
        days = ret_col.split("_")[1]
        summary = summarize_backtest(bt, "signal", ret_col)
        if summary.empty:
            continue
        for sig_type, row in summary.iterrows():
            if row["count"] >= 3 and row["avg_return"] > 1.5 and row["win_rate"] > 55:
                notes.append({
                    "type": "bullish_pattern",
                    "importance": "HIGH",
                    "title": f"'{sig_type}' 시그널 유효 ({days} 기준)",
                    "body": (f"과거 {int(row['count'])}건 중 승률 {row['win_rate']:.0f}%, "
                             f"평균 수익률 +{row['avg_return']:.1f}%"),
                })
            elif row["count"] >= 3 and row["avg_return"] < -1.5 and row["win_rate"] < 45:
                notes.append({
                    "type": "bearish_pattern",
                    "importance": "HIGH",
                    "title": f"'{sig_type}' 시그널 역행 ({days} 기준)",
                    "body": (f"과거 {int(row['count'])}건 중 승률 {row['win_rate']:.0f}%, "
                             f"평균 수익률 {row['avg_return']:.1f}%"),
                })

    # 2) Recent signals with good track record
    recent = get_current_signals(signals_df, lookback_days=7)
    if not recent.empty:
        for _, sig in recent.iterrows():
            sig_summary = summarize_backtest(
                bt[bt["signal"] == sig["signal"]], "signal", "return_5d")
            if not sig_summary.empty:
                sr = sig_summary.iloc[0]
                if sr["win_rate"] > 55 and sr["avg_return"] > 1:
                    notes.append({
                        "type": "action",
                        "importance": "MEDIUM",
                        "title": f"주목: {sig['name']} ({sig['signal']})",
                        "body": (f"{sig['etf']}가 {sig['detail']}. "
                                 f"이 시그널 과거 승률 {sr['win_rate']:.0f}%"),
                    })

    return notes


# ── Walk-Forward Backtest (Out-of-Sample) ────────────────────────────────────

def walk_forward_backtest(tf_df: pd.DataFrame, ko_df: pd.DataFrame,
                          prices: dict, ticker_map: dict,
                          hold_days: int = 10,
                          train_window_days: int = 60,
                          step_days: int = 5) -> pd.DataFrame:
    """Walk-forward backtest: at each rebalance date, look at signals from the
    past train_window and decide which to trade, then measure actual forward return.

    This avoids lookahead bias — we only use data available up to each decision point.

    Strategy: At each step, if a BIG_INCREASE or CONVICTION_BUY signal fired in the
    last `step_days` days, buy that stock and hold for `hold_days`.
    Compare vs. doing nothing (benchmark = 0).
    """
    if tf_df.empty and ko_df.empty:
        return pd.DataFrame()

    # Get all signals
    all_signals = pd.concat([
        detect_signals(tf_df, "Timefolio") if not tf_df.empty else pd.DataFrame(),
        detect_signals(ko_df, "Koact") if not ko_df.empty else pd.DataFrame(),
    ], ignore_index=True)

    if all_signals.empty:
        return pd.DataFrame()

    all_dates = sorted(all_signals["date"].unique())
    if len(all_dates) < 20:
        return pd.DataFrame()

    # Start walk-forward from train_window days after first signal
    start_idx = 0
    for i, d in enumerate(all_dates):
        if (pd.Timestamp(d) - pd.Timestamp(all_dates[0])).days >= train_window_days:
            start_idx = i
            break

    results = []
    decision_dates = all_dates[start_idx::max(1, step_days // 2)]

    for decision_date in decision_dates:
        dt = pd.Timestamp(decision_date)

        # Look at signals in the recent window (last step_days)
        window_start = dt - pd.Timedelta(days=step_days)
        recent_sigs = all_signals[
            (all_signals["date"] > window_start) &
            (all_signals["date"] <= dt)
        ]

        if recent_sigs.empty:
            continue

        # Strategy: trade BIG_INCREASE and CONVICTION_BUY signals
        buy_sigs = recent_sigs[recent_sigs["signal"].isin(
            ["BIG_INCREASE", "CONVICTION_BUY", "NEW_ENTRY"])]
        sell_sigs = recent_sigs[recent_sigs["signal"].isin(["EXIT"])]

        # Compute actual returns for buy signals
        for _, sig in buy_sigs.iterrows():
            norm = sig["norm_name"]
            if norm not in prices:
                continue
            price_df = prices[norm]
            price_df_dt = price_df.copy()
            price_df_dt["Date"] = pd.to_datetime(price_df_dt["Date"])

            entry_prices = price_df_dt[price_df_dt["Date"] >= dt].head(1)
            if entry_prices.empty:
                continue
            entry_p = float(entry_prices.iloc[0]["Close"])
            entry_d = entry_prices.iloc[0]["Date"]

            exit_target = entry_d + pd.Timedelta(days=hold_days)
            exit_prices = price_df_dt[price_df_dt["Date"] >= exit_target].head(1)
            if exit_prices.empty:
                continue
            exit_p = float(exit_prices.iloc[0]["Close"])
            ret = (exit_p - entry_p) / entry_p * 100

            results.append({
                "decision_date": decision_date,
                "action": "BUY",
                "signal": sig["signal"],
                "etf": sig["etf"],
                "name": sig["name"],
                "norm_name": norm,
                "entry_date": entry_d,
                "exit_date": exit_prices.iloc[0]["Date"],
                "entry_price": entry_p,
                "exit_price": exit_p,
                "return_pct": ret,
                "hold_days": hold_days,
            })

        # Short / avoid signals for EXIT
        for _, sig in sell_sigs.iterrows():
            norm = sig["norm_name"]
            if norm not in prices:
                continue
            price_df = prices[norm]
            price_df_dt = price_df.copy()
            price_df_dt["Date"] = pd.to_datetime(price_df_dt["Date"])

            entry_prices = price_df_dt[price_df_dt["Date"] >= dt].head(1)
            if entry_prices.empty:
                continue
            entry_p = float(entry_prices.iloc[0]["Close"])
            entry_d = entry_prices.iloc[0]["Date"]

            exit_target = entry_d + pd.Timedelta(days=hold_days)
            exit_prices = price_df_dt[price_df_dt["Date"] >= exit_target].head(1)
            if exit_prices.empty:
                continue
            exit_p = float(exit_prices.iloc[0]["Close"])
            # For EXIT signals, "success" = stock went DOWN (we avoided/shorted)
            ret = (entry_p - exit_p) / entry_p * 100  # positive = stock dropped = good call

            results.append({
                "decision_date": decision_date,
                "action": "AVOID/SHORT",
                "signal": sig["signal"],
                "etf": sig["etf"],
                "name": sig["name"],
                "norm_name": norm,
                "entry_date": entry_d,
                "exit_date": exit_prices.iloc[0]["Date"],
                "entry_price": entry_p,
                "exit_price": exit_p,
                "return_pct": ret,
                "hold_days": hold_days,
            })

    return pd.DataFrame(results)


def compute_equity_curve(wf_results: pd.DataFrame) -> pd.DataFrame:
    """From walk-forward results, compute an equity curve assuming equal-weight
    allocation across all signals on each decision date."""
    if wf_results.empty:
        return pd.DataFrame()

    # Group by decision_date, compute average return
    by_date = wf_results.groupby("decision_date").agg(
        avg_return=("return_pct", "mean"),
        n_trades=("return_pct", "count"),
        win_rate=("return_pct", lambda x: (x > 0).mean() * 100),
    ).reset_index()
    by_date = by_date.sort_values("decision_date")

    # Cumulative equity
    by_date["cumulative"] = (1 + by_date["avg_return"] / 100).cumprod() * 100 - 100
    by_date["cumulative_factor"] = (1 + by_date["avg_return"] / 100).cumprod()

    return by_date


# ── Full Analysis Pipeline ───────────────────────────────────────────────────

def run_full_analysis(tf_df: pd.DataFrame, ko_df: pd.DataFrame) -> dict:
    """Run the complete signal analysis pipeline."""
    # 1. Detect signals
    tf_signals = detect_signals(tf_df, "Timefolio") if not tf_df.empty else pd.DataFrame()
    ko_signals = detect_signals(ko_df, "Koact") if not ko_df.empty else pd.DataFrame()
    all_signals = pd.concat([tf_signals, ko_signals], ignore_index=True)

    # 2. Build ticker map and fetch prices
    ticker_map = build_ticker_map(tf_df, ko_df)

    dates = []
    for df in [tf_df, ko_df]:
        if not df.empty:
            dates.extend(get_dates(df))
    if dates:
        start = min(dates)
        end = max(dates)
        start_date = pd.Timestamp(start).date()
        end_date = pd.Timestamp(end).date()
    else:
        start_date = date.today() - timedelta(days=180)
        end_date = date.today()

    prices = fetch_all_prices(ticker_map, start_date, end_date)

    # 3. Backtest
    bt = backtest_signals(all_signals, prices)

    # 4. Consensus & divergence
    consensus = find_consensus_signals(tf_signals, ko_signals)
    divergence = find_divergence_signals(tf_df, ko_df)

    # 5. Generate insights
    notes = generate_trading_notes(bt, all_signals)

    return {
        "signals": all_signals,
        "backtest": bt,
        "consensus": consensus,
        "divergence": divergence,
        "notes": notes,
        "ticker_map": ticker_map,
        "prices": prices,
        "tf_signals": tf_signals,
        "ko_signals": ko_signals,
    }
