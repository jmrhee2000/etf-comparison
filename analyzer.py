"""ETF 포트폴리오 분석 엔진 — 위너/루저, 편입/편출 추적"""
from datetime import date
import pandas as pd
from collector import load_all_data


def _normalize_name(name: str) -> str:
    """Normalize stock name for matching across ETFs."""
    name = name.strip().upper()
    # Remove common suffixes
    for suffix in [" INC", " CORP", " LTD", " PLC", " CO", " CLASS A",
                   " CL A", "-CL A", "-CL C", " CLASS C", "/DE"]:
        name = name.replace(suffix, "")
    return name.strip()


# ── Data Loading ─────────────────────────────────────────────────────────────

def get_portfolio_data(etf: str) -> pd.DataFrame:
    """Load and clean portfolio data for an ETF."""
    df = load_all_data(etf)
    if df.empty:
        return df
    # Filter out cash entries
    cash_names = {"현금", "설정현금액", "원화현금", "CASH"}
    df = df[~df["name"].str.strip().isin(cash_names)].copy()
    df = df[df["weight"] > 0].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "weight"], ascending=[True, False])
    df["norm_name"] = df["name"].apply(_normalize_name)
    return df


def get_dates(df: pd.DataFrame) -> list:
    """Get sorted unique dates."""
    return sorted(df["date"].unique())


# ── Entry/Exit Tracking ─────────────────────────────────────────────────────

def track_entries_exits(df: pd.DataFrame) -> pd.DataFrame:
    """Track stock entry/exit events.

    Returns DataFrame with columns:
        name, norm_name, event (entry/exit), date, weight_at_event
    """
    if df.empty:
        return pd.DataFrame()

    dates = get_dates(df)
    events = []

    for i, d in enumerate(dates):
        current_stocks = set(df[df["date"] == d]["norm_name"])
        if i == 0:
            # First date — all stocks are "entry"
            for _, row in df[df["date"] == d].iterrows():
                events.append({
                    "name": row["name"],
                    "norm_name": row["norm_name"],
                    "event": "entry",
                    "date": d,
                    "weight": row["weight"],
                })
            prev_stocks = current_stocks
            continue

        prev_stocks_set = set(df[df["date"] == dates[i-1]]["norm_name"])

        # New entries
        new = current_stocks - prev_stocks_set
        for _, row in df[(df["date"] == d) & (df["norm_name"].isin(new))].iterrows():
            events.append({
                "name": row["name"],
                "norm_name": row["norm_name"],
                "event": "entry",
                "date": d,
                "weight": row["weight"],
            })

        # Exits
        exited = prev_stocks_set - current_stocks
        for norm in exited:
            prev_row = df[(df["date"] == dates[i-1]) & (df["norm_name"] == norm)]
            if not prev_row.empty:
                events.append({
                    "name": prev_row.iloc[0]["name"],
                    "norm_name": norm,
                    "event": "exit",
                    "date": d,
                    "weight": prev_row.iloc[0]["weight"],
                })

    return pd.DataFrame(events)


def get_holding_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate holding periods for each stock.

    Returns DataFrame with: name, entry_date, exit_date, duration_days,
                            entry_weight, exit_weight, max_weight, min_weight, still_held
    """
    if df.empty:
        return pd.DataFrame()

    events = track_entries_exits(df)
    if events.empty:
        return pd.DataFrame()

    dates = get_dates(df)
    last_date = dates[-1]
    results = []

    for norm_name in events["norm_name"].unique():
        stock_events = events[events["norm_name"] == norm_name].sort_values("date")
        stock_data = df[df["norm_name"] == norm_name]
        display_name = stock_data.iloc[-1]["name"]

        entries = stock_events[stock_events["event"] == "entry"]["date"].tolist()
        exits = stock_events[stock_events["event"] == "exit"]["date"].tolist()

        for entry_date in entries:
            # Find matching exit (first exit after this entry)
            matching_exits = [e for e in exits if e > entry_date]
            if matching_exits:
                exit_date = matching_exits[0]
                still_held = False
            else:
                exit_date = last_date
                still_held = True

            # Weight data during holding period
            mask = (stock_data["date"] >= entry_date) & (stock_data["date"] <= exit_date)
            period_data = stock_data[mask]

            if period_data.empty:
                continue

            entry_weight = period_data.iloc[0]["weight"]
            exit_weight = period_data.iloc[-1]["weight"]

            results.append({
                "name": display_name,
                "norm_name": norm_name,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "duration_days": (pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days,
                "entry_weight": entry_weight,
                "exit_weight": exit_weight,
                "max_weight": period_data["weight"].max(),
                "min_weight": period_data["weight"].min(),
                "still_held": still_held,
            })

    return pd.DataFrame(results)


# ── Weight Change Tracking ───────────────────────────────────────────────────

def get_weight_timeseries(df: pd.DataFrame, norm_name: str) -> pd.DataFrame:
    """Get daily weight timeseries for a stock."""
    stock = df[df["norm_name"] == norm_name][["date", "weight", "quantity"]].copy()
    return stock.sort_values("date")


def get_all_weight_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot: dates as rows, stocks as columns, weights as values."""
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot_table(index="date", columns="name", values="weight", fill_value=0)
    return pivot.sort_index()


# ── Winner / Loser Classification ────────────────────────────────────────────

def classify_winners_losers(df: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
    """Classify stocks as winners or losers based on weight AND quantity changes.

    Logic:
    - weight_change > 0 + qty_change >= 0 → 'winner' (price went up)
    - weight_change < 0 + qty_change >= 0 → 'loser' (price went down)
    - qty_change detected → 'active_increase' or 'active_decrease' (manager action)
    """
    if df.empty:
        return pd.DataFrame()

    dates = get_dates(df)
    if len(dates) < 2:
        return pd.DataFrame()

    end_date = dates[-1]
    start_date = pd.Timestamp(end_date) - pd.Timedelta(days=lookback_days)
    # Find the closest actual date
    start_candidates = [d for d in dates if pd.Timestamp(d) >= start_date]
    if not start_candidates:
        start_candidates = dates
    actual_start = start_candidates[0]

    start_data = df[df["date"] == actual_start].set_index("norm_name")
    end_data = df[df["date"] == end_date].set_index("norm_name")

    all_stocks = set(start_data.index) | set(end_data.index)
    results = []

    for norm in all_stocks:
        in_start = norm in start_data.index
        in_end = norm in end_data.index

        if in_start and in_end:
            s = start_data.loc[norm]
            e = end_data.loc[norm]
            name = e["name"] if isinstance(e, pd.Series) else e.iloc[0]["name"]
            w_start = float(s["weight"]) if isinstance(s, pd.Series) else float(s.iloc[0]["weight"])
            w_end = float(e["weight"]) if isinstance(e, pd.Series) else float(e.iloc[0]["weight"])
            q_start = float(s["quantity"]) if isinstance(s, pd.Series) else float(s.iloc[0]["quantity"])
            q_end = float(e["quantity"]) if isinstance(e, pd.Series) else float(e.iloc[0]["quantity"])

            w_change = w_end - w_start
            q_change = q_end - q_start
            q_pct = (q_change / q_start * 100) if q_start > 0 else 0

            # Classification
            if abs(q_pct) < 5:  # quantity barely changed → price-driven
                category = "winner" if w_change > 0.1 else ("loser" if w_change < -0.1 else "neutral")
            else:
                category = "active_increase" if q_change > 0 else "active_decrease"

            results.append({
                "name": name,
                "norm_name": norm,
                "weight_start": w_start,
                "weight_end": w_end,
                "weight_change": w_change,
                "qty_start": q_start,
                "qty_end": q_end,
                "qty_change_pct": q_pct,
                "category": category,
                "status": "held",
            })
        elif in_start and not in_end:
            s = start_data.loc[norm]
            name = s["name"] if isinstance(s, pd.Series) else s.iloc[0]["name"]
            w_start = float(s["weight"]) if isinstance(s, pd.Series) else float(s.iloc[0]["weight"])
            results.append({
                "name": name,
                "norm_name": norm,
                "weight_start": w_start,
                "weight_end": 0,
                "weight_change": -w_start,
                "qty_start": float(s["quantity"]) if isinstance(s, pd.Series) else float(s.iloc[0]["quantity"]),
                "qty_end": 0,
                "qty_change_pct": -100,
                "category": "exited",
                "status": "exited",
            })
        else:  # not in_start, in_end → new entry
            e = end_data.loc[norm]
            name = e["name"] if isinstance(e, pd.Series) else e.iloc[0]["name"]
            w_end = float(e["weight"]) if isinstance(e, pd.Series) else float(e.iloc[0]["weight"])
            results.append({
                "name": name,
                "norm_name": norm,
                "weight_start": 0,
                "weight_end": w_end,
                "weight_change": w_end,
                "qty_start": 0,
                "qty_end": float(e["quantity"]) if isinstance(e, pd.Series) else float(e.iloc[0]["quantity"]),
                "qty_change_pct": 100,
                "category": "new_entry",
                "status": "new_entry",
            })

    return pd.DataFrame(results).sort_values("weight_change", ascending=False)


# ── Cross-ETF Comparison ────────────────────────────────────────────────────

def compare_holdings(tf_df: pd.DataFrame, ko_df: pd.DataFrame,
                     as_of: pd.Timestamp | None = None) -> dict:
    """Compare holdings between two ETFs on a given date."""
    if tf_df.empty or ko_df.empty:
        return {"common": [], "timefolio_only": [], "koact_only": []}

    if as_of is None:
        tf_dates = get_dates(tf_df)
        ko_dates = get_dates(ko_df)
        as_of = min(tf_dates[-1], ko_dates[-1])

    tf_latest = tf_df[tf_df["date"] == as_of]
    ko_latest = ko_df[ko_df["date"] == as_of]

    # If exact date not found, find closest
    if tf_latest.empty:
        tf_dates_arr = sorted(tf_df["date"].unique())
        closest = min(tf_dates_arr, key=lambda d: abs(pd.Timestamp(d) - pd.Timestamp(as_of)))
        tf_latest = tf_df[tf_df["date"] == closest]
    if ko_latest.empty:
        ko_dates_arr = sorted(ko_df["date"].unique())
        closest = min(ko_dates_arr, key=lambda d: abs(pd.Timestamp(d) - pd.Timestamp(as_of)))
        ko_latest = ko_df[ko_df["date"] == closest]

    tf_stocks = set(tf_latest["norm_name"])
    ko_stocks = set(ko_latest["norm_name"])

    common = tf_stocks & ko_stocks
    tf_only = tf_stocks - ko_stocks
    ko_only = ko_stocks - tf_stocks

    common_details = []
    for norm in common:
        tf_row = tf_latest[tf_latest["norm_name"] == norm].iloc[0]
        ko_row = ko_latest[ko_latest["norm_name"] == norm].iloc[0]
        common_details.append({
            "name": tf_row["name"],
            "norm_name": norm,
            "tf_weight": tf_row["weight"],
            "ko_weight": ko_row["weight"],
            "weight_diff": tf_row["weight"] - ko_row["weight"],
        })

    return {
        "common": sorted(common_details, key=lambda x: -max(x["tf_weight"], x["ko_weight"])),
        "timefolio_only": [
            {"name": tf_latest[tf_latest["norm_name"] == n].iloc[0]["name"],
             "weight": tf_latest[tf_latest["norm_name"] == n].iloc[0]["weight"]}
            for n in tf_only
        ],
        "koact_only": [
            {"name": ko_latest[ko_latest["norm_name"] == n].iloc[0]["name"],
             "weight": ko_latest[ko_latest["norm_name"] == n].iloc[0]["weight"]}
            for n in ko_only
        ],
    }


def compare_entry_exit_timing(tf_df: pd.DataFrame, ko_df: pd.DataFrame) -> pd.DataFrame:
    """Compare when each ETF entered/exited common stocks."""
    tf_events = track_entries_exits(tf_df)
    ko_events = track_entries_exits(ko_df)

    if tf_events.empty or ko_events.empty:
        return pd.DataFrame()

    # Find stocks that appear in both
    common_norms = set(tf_events["norm_name"]) & set(ko_events["norm_name"])
    results = []

    for norm in common_norms:
        tf_entries = tf_events[(tf_events["norm_name"] == norm) & (tf_events["event"] == "entry")]
        ko_entries = ko_events[(ko_events["norm_name"] == norm) & (ko_events["event"] == "entry")]

        tf_first_entry = tf_entries["date"].min() if not tf_entries.empty else None
        ko_first_entry = ko_entries["date"].min() if not ko_entries.empty else None

        if tf_first_entry is not None and ko_first_entry is not None:
            diff = (pd.Timestamp(tf_first_entry) - pd.Timestamp(ko_first_entry)).days
            leader = "Timefolio" if diff < 0 else ("Koact" if diff > 0 else "Same")
        else:
            diff = None
            leader = "N/A"

        name = tf_entries.iloc[0]["name"] if not tf_entries.empty else ko_entries.iloc[0]["name"]
        results.append({
            "name": name,
            "norm_name": norm,
            "tf_first_entry": tf_first_entry,
            "ko_first_entry": ko_first_entry,
            "entry_diff_days": diff,
            "earlier_entry": leader,
        })

    return pd.DataFrame(results).sort_values("entry_diff_days", key=abs, na_position="last")
