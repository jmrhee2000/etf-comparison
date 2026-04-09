import os
from datetime import date, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

TIMEFOLIO = {
    "name": "타임폴리오 글로벌AI",
    "short": "Timefolio",
    "url": "https://timeetf.co.kr/m11_view.php",
    "idx": 6,
    "data_dir": os.path.join(DATA_DIR, "timefolio"),
}

KOACT = {
    "name": "Koact 미국나스닥성장기업",
    "short": "Koact",
    "url": "https://www.samsungactive.co.kr/api/v1/product/etf-pdf/2ETFQ1.do",
    "fId": "2ETFQ1",
    "data_dir": os.path.join(DATA_DIR, "koact"),
}

REQUEST_DELAY = 0.5  # seconds between requests
DEFAULT_LOOKBACK_MONTHS = 6

def get_default_start_date():
    today = date.today()
    return today - timedelta(days=DEFAULT_LOOKBACK_MONTHS * 30)
