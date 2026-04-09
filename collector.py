"""ETF 포트폴리오 일별 데이터 수집기"""
import os
import re
import time
import json
import logging
from datetime import date, timedelta, datetime

import requests
import pandas as pd
from bs4 import BeautifulSoup

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
})


# ── Timefolio ────────────────────────────────────────────────────────────────

def fetch_timefolio(d: date) -> pd.DataFrame | None:
    """Fetch Timefolio portfolio for a given date. Returns DataFrame or None."""
    url = f"{config.TIMEFOLIO['url']}?idx={config.TIMEFOLIO['idx']}&pdfDate={d.isoformat()}"
    try:
        resp = SESSION.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning("Timefolio request failed for %s: %s", d, e)
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.select_one("table.table3.moreList1")
    if not table:
        log.warning("Timefolio: table not found for %s", d)
        return None

    rows = []
    for tr in table.select("tbody tr"):
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue
        ticker = tds[0].get_text(strip=True)
        name = tds[1].get_text(strip=True)
        qty_str = tds[2].get_text(strip=True).replace(",", "")
        eval_str = tds[3].get_text(strip=True).replace(",", "")
        weight_str = tds[4].get_text(strip=True)
        try:
            qty = float(qty_str) if qty_str else 0
            eval_amt = float(eval_str) if eval_str else 0
            weight = float(weight_str) if weight_str else 0
        except ValueError:
            continue
        rows.append({
            "ticker": ticker,
            "name": name,
            "quantity": qty,
            "eval_amount": eval_amt,
            "weight": weight,
        })

    if not rows:
        log.info("Timefolio: no data for %s (holiday?)", d)
        return None

    df = pd.DataFrame(rows)
    df["date"] = d.isoformat()
    return df


# ── Koact ────────────────────────────────────────────────────────────────────

def _new_koact_session() -> requests.Session:
    """Create a fresh session for Koact requests to avoid rate-limit state."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.samsungactive.co.kr/etf/view.do?id=2ETFQ1",
    })
    # Visit the main page first to get cookies
    try:
        s.get("https://www.samsungactive.co.kr/etf/view.do?id=2ETFQ1", timeout=10)
    except Exception:
        pass
    return s

_koact_session = None
_koact_req_count = 0

def fetch_koact(d: date) -> pd.DataFrame | None:
    """Fetch Koact portfolio for a given date. Returns DataFrame or None."""
    global _koact_session, _koact_req_count

    # Reset session every 20 requests to avoid rate limiting
    if _koact_session is None or _koact_req_count >= 20:
        _koact_session = _new_koact_session()
        _koact_req_count = 0
        time.sleep(2)

    ymd = d.strftime("%Y%m%d")
    url = f"{config.KOACT['url']}?gijunYMD={ymd}"
    try:
        resp = _koact_session.get(url, timeout=15)
        _koact_req_count += 1
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        log.warning("Koact request failed for %s: %s", d, e)
        # If 403, reset session immediately
        if hasattr(e, 'response') and getattr(e, 'response', None) is not None and e.response.status_code == 403:
            _koact_session = None
            _koact_req_count = 0
        return None

    pdf = data.get("pdf", {})
    actual_ymd = pdf.get("gijunYMD", "")
    items = pdf.get("list", [])
    if not items:
        log.info("Koact: no data for %s", d)
        return None

    # API returns closest business day — check if it matches requested date
    if actual_ymd:
        actual_date = datetime.strptime(actual_ymd, "%Y%m%d").date()
    else:
        actual_date = d

    rows = []
    for item in items:
        name = item.get("secNm", "")
        ticker = item.get("itmNo", "")
        ratio_str = item.get("ratio", "")
        qty_str = item.get("applyQ", "")
        eval_str = item.get("evalA", "")

        # Skip cash/설정현금액 entries without ratio
        if not ratio_str and name in ("설정현금액", "원화현금"):
            # Still include cash but with 0 ratio
            pass

        try:
            weight = float(ratio_str) if ratio_str else 0
            qty = float(qty_str) if qty_str else 0
            eval_amt = float(eval_str) if eval_str else 0
        except ValueError:
            continue

        rows.append({
            "ticker": ticker,
            "name": name,
            "quantity": qty,
            "eval_amount": eval_amt,
            "weight": weight,
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["date"] = actual_date.isoformat()
    return df


# ── Collection orchestration ─────────────────────────────────────────────────

def _business_days(start: date, end: date) -> list[date]:
    """Generate weekday dates between start and end inclusive."""
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri
            days.append(d)
        d += timedelta(days=1)
    return days


def _save_csv(df: pd.DataFrame, data_dir: str, d: date):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{d.isoformat()}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _already_collected(data_dir: str, d: date) -> bool:
    path = os.path.join(data_dir, f"{d.isoformat()}.csv")
    return os.path.exists(path)


def collect_etf(etf_name: str, fetch_fn, data_dir: str,
                start: date, end: date, force: bool = False):
    """Collect daily portfolio data for one ETF."""
    days = _business_days(start, end)
    total = len(days)
    collected = 0
    skipped = 0
    failed = 0
    seen_dates = set()

    for i, d in enumerate(days):
        if not force and _already_collected(data_dir, d):
            skipped += 1
            continue

        df = fetch_fn(d)
        if df is not None:
            actual_date_str = df["date"].iloc[0]
            actual_date = date.fromisoformat(actual_date_str)
            if actual_date not in seen_dates:
                _save_csv(df, data_dir, actual_date)
                seen_dates.add(actual_date)
                collected += 1
                log.info("[%s] %d/%d collected %s (actual: %s)",
                         etf_name, i+1, total, d, actual_date)
            else:
                skipped += 1
        else:
            failed += 1

        delay = 1.5 if fetch_fn == fetch_koact else config.REQUEST_DELAY
        time.sleep(delay)

    log.info("[%s] Done: %d collected, %d skipped, %d failed (of %d days)",
             etf_name, collected, skipped, failed, total)
    return collected


def collect_all(start: date | None = None, end: date | None = None, force: bool = False):
    """Collect data for both ETFs."""
    if start is None:
        start = config.get_default_start_date()
    if end is None:
        end = date.today()

    log.info("Collecting from %s to %s", start, end)

    log.info("=== Timefolio ===")
    collect_etf("Timefolio", fetch_timefolio,
                config.TIMEFOLIO["data_dir"], start, end, force)

    log.info("=== Koact ===")
    collect_etf("Koact", fetch_koact,
                config.KOACT["data_dir"], start, end, force)


def update():
    """Incremental update — collect only new data since last collection."""
    for etf_key, data_dir_key in [("TIMEFOLIO", "data_dir"), ("KOACT", "data_dir")]:
        etf_conf = getattr(config, etf_key)
        data_dir = etf_conf[data_dir_key]
        if os.path.exists(data_dir):
            files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
            if files:
                last_date = date.fromisoformat(files[-1].replace(".csv", ""))
                start = last_date + timedelta(days=1)
            else:
                start = config.get_default_start_date()
        else:
            start = config.get_default_start_date()

        end = date.today()
        if start > end:
            log.info("[%s] Already up to date", etf_conf["short"])
            continue

        fetch_fn = fetch_timefolio if etf_key == "TIMEFOLIO" else fetch_koact
        collect_etf(etf_conf["short"], fetch_fn, data_dir, start, end)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_all_data(etf: str) -> pd.DataFrame:
    """Load all collected CSV files for an ETF into a single DataFrame."""
    if etf == "timefolio":
        data_dir = config.TIMEFOLIO["data_dir"]
    elif etf == "koact":
        data_dir = config.KOACT["data_dir"]
    else:
        raise ValueError(f"Unknown ETF: {etf}")

    if not os.path.exists(data_dir):
        return pd.DataFrame()

    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        path = os.path.join(data_dir, f)
        df = pd.read_csv(path, encoding="utf-8-sig")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.date
    return combined


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "update":
        update()
    else:
        collect_all()
