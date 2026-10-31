# backend/app.py
import asyncio
import json
import os
import math
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd

from tarkogu.get_data import get_item_metadata, refresh_item_historical_price_cache, UPDATE_INTERVAL
from tarkogu.predict import predict_price_and_offer_count, find_buy_times, find_sell_times
from tarkogu.utils import get_pkg_abs_path

PRED_CATEGORIES = {'Barter item', 'Item', 'Food and drink', 'Key', 'Equipment'}
REF_DAYS = 21
PREDICT_HOURS = 24

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kusurin-tech.icu",
        "https://www.kusurin-tech.icu",
        "https://kusurin.icu",
        "https://www.kusurin.icu",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_PATH = Path(get_pkg_abs_path()) / "data" / "rankings_cache.json"

W_SORT = {"buy_range": 0.0, "sell_range": 0.0, "rmse": -1.0}

DEFAULT_CACHE = {"updated_at": None, "items": []}
NUMERIC_FIELDS = {"score", "buy_range", "sell_range", "rmse", "offer_yhat_mean_24h"}
CACHE_LOCK = Lock()

def save_cache(data: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CACHE_PATH.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp_path, CACHE_PATH)

def load_cache() -> dict:
    try:
        stamp = CACHE_PATH.stat()
    except OSError:
        return DEFAULT_CACHE.copy()
    with CACHE_LOCK:
        return read_cache(stamp.st_mtime_ns, stamp.st_size)


@lru_cache(maxsize=1)
def read_cache(mtime_ns: int, size: int) -> dict:
    # Reuse one snapshot across page requests until the atomic cache changes.
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return DEFAULT_CACHE.copy()
    if not isinstance(data, dict):
        return DEFAULT_CACHE.copy()
    items = data.get("items", [])
    items = sanitize_items(items) if isinstance(items, list) else []
    return {
        "updated_at": data.get("updated_at"),
        "items": items,
    }

def normalize_number(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def serialize_time_series(df: pd.DataFrame, time_col: str, value_col: str) -> list:
    if df.empty:
        return []
    series = []
    for _, row in df.iterrows():
        timestamp = row[time_col]
        if pd.isna(timestamp):
            continue
        value = normalize_number(row[value_col])
        if value is None:
            continue
        if isinstance(timestamp, pd.Timestamp):
            time_str = timestamp.isoformat()
        else:
            time_str = pd.to_datetime(timestamp).isoformat()
        series.append({"time": time_str, value_col: value})
    return series

def serialize_time_list(values: list) -> list:
    times = []
    for value in values:
        if pd.isna(value):
            continue
        if isinstance(value, pd.Timestamp):
            times.append(value.isoformat())
        else:
            times.append(pd.to_datetime(value).isoformat())
    return times

def mean_offer_next_24h(df: pd.DataFrame, time_col: str, value_col: str, now: pd.Timestamp) -> float | None:
    if df.empty:
        return None
    window_end = now + pd.Timedelta(hours=24)
    window = df[(df[time_col] >= now) & (df[time_col] <= window_end)]
    if window.empty:
        return None
    return normalize_number(window[value_col].mean())

def sanitize_item(item: dict) -> dict:
    sanitized = dict(item)
    for field in NUMERIC_FIELDS:
        if field in sanitized:
            sanitized[field] = normalize_number(sanitized[field])
    return sanitized

def sanitize_items(items: list) -> list:
    sanitized_items = []
    for item in items:
        if isinstance(item, dict):
            sanitized_items.append(sanitize_item(item))
    return sanitized_items

def build_items_by_id(cache_items: list) -> dict:
    items_by_id = {}
    for item in cache_items:
        item_id = item.get("id") if isinstance(item, dict) else None
        if item_id:
            items_by_id[item_id] = item
    return items_by_id

def update_loop():
    while True:
        cycle_start = time.monotonic()
        sleep_seconds = None

        try:
            meta_df = get_item_metadata()
            meta_df = meta_df[meta_df["category"].isin(PRED_CATEGORIES)]

            items = meta_df.index.tolist()
            if not items:
                sleep_seconds = 60
                continue

            cache = load_cache()
            items_by_id = build_items_by_id(cache.get("items", []))

            for item_number, item_id in enumerate(items, start=1):
                if (item_number - 1) % 20 == 0:
                    price_by_id = refresh_item_historical_price_cache(
                        items[item_number - 1:item_number + 19], REF_DAYS)
                price_df = price_by_id.get(item_id)
                if price_df is None or price_df.empty:
                    print(f"No data for {item_id}")
                    continue

                try:
                    res, rmse = predict_price_and_offer_count(
                        item_id,
                        REF_DAYS,
                        PREDICT_HOURS,
                        price_df=price_df,
                    )
                    buy_time, buy_range = find_buy_times(res)
                    sell_time, sell_range = find_sell_times(res)

                    score = W_SORT['rmse'] * rmse / buy_range if buy_range > 0 else None

                    item_meta = meta_df.loc[item_id] if item_id in meta_df.index else None
                    name = item_meta["name"] if item_meta is not None and "name" in item_meta else None
                    name_zh = item_meta["name_zh"] if item_meta is not None and "name_zh" in item_meta else None
                    icon_link = item_meta["iconLink"] if item_meta is not None and "iconLink" in item_meta else None
                    category = item_meta["category"] if item_meta is not None and "category" in item_meta else None

                    now = pd.Timestamp.now(timezone.utc).tz_localize(None)
                    window_start = now - pd.Timedelta(hours=48)
                    window_end = now + pd.Timedelta(hours=24)

                    price_window = res[(res["price_ds"] >= window_start) & (res["price_ds"] <= window_end)]
                    price_series = serialize_time_series(price_window, "price_ds", "price_yhat")
                    price_df = price_df[price_df["ds"] >= now - pd.Timedelta(hours=48)]
                    historical_price_series = serialize_time_series(price_df, "ds", "y")

                    offer_mean_24h = mean_offer_next_24h(res, "offer_ds", "offer_yhat", now)

                    items_by_id[item_id] = sanitize_item({
                        "id": item_id,
                        "name": name,
                        "name_zh": name_zh,
                        "icon_link": icon_link,
                        "category": category,
                        "score": score,
                        "buy_range": buy_range,
                        "sell_range": sell_range,
                        "rmse": rmse,
                        "buy_time": serialize_time_list(buy_time),
                        "sell_time": serialize_time_list(sell_time),
                        "historical_price_series": historical_price_series,
                        "price_yhat_series": price_series,
                        "offer_yhat_mean_24h": offer_mean_24h,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    })
                    # Publish progress on a fresh VPS instead of waiting for
                    # every model to finish before exposing the first page.
                    if item_number % 20 == 0 or len(items_by_id) == 1:
                        save_cache({
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                            "items": sorted(items_by_id.values(), key=lambda item:
                                item["score"] if item.get("score") is not None else -math.inf,
                                reverse=True),
                        })
                except Exception as exc:
                    print(f"{datetime.now(timezone.utc).isoformat()} err on {item_id}: {exc}")

            items_sorted = sorted(
                items_by_id.values(),
                key=lambda item: item.get("score") if item.get("score") is not None else float("-9999"),
                reverse=True,
            )

            save_cache({
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "items": sanitize_items(items_sorted),
            })
        except Exception as exc:
            print(f"{datetime.now(timezone.utc).isoformat()} update loop error: {exc}")
            sleep_seconds = 60
        finally:
            if sleep_seconds is None:
                elapsed = time.monotonic() - cycle_start
                sleep_seconds = max(0, UPDATE_INTERVAL / 1000 - elapsed)
            if sleep_seconds > 0:
                time.sleep(max(1, sleep_seconds))

@app.on_event("startup")
async def on_startup():
    # Fetching data and fitting Prophet must not block pagination requests.
    asyncio.create_task(asyncio.to_thread(update_loop))


def item_key(item: dict) -> str:
    item_id = item.get("id") or item.get("item_id") or item.get("itemId")
    if item_id:
        return f"id:{item_id}"
    name_en = (item.get("name_en") or item.get("name_en_us") or
               item.get("name_en_gb") or item.get("name") or "")
    return f"name:{str(item.get('name_zh') or '').strip()}|{str(name_en).strip()}"


def match_item(item: dict, query: str) -> bool:
    names = " ".join(str(item.get(field) or "") for field in
                     ("name_zh", "name", "name_en", "name_en_us", "name_en_gb"))
    return query in names.casefold()

@app.get("/rankings")
@app.get("/api/rankings")
def get_rankings(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    q: str = Query("", max_length=200),
    pinned: list[str] | None = Query(None),
):
    cache = load_cache()
    query = q.strip().casefold()
    items = [item for item in cache["items"] if not query or match_item(item, query)]
    # Stable partition preserves ranking order inside each pin group.
    pinned_keys = set(pinned or [])
    if pinned_keys:
        items.sort(key=lambda item: item_key(item) not in pinned_keys)
    total = len(items)
    total_pages = math.ceil(total / page_size)
    page = min(page, max(1, total_pages))
    offset = (page - 1) * page_size
    return {
        "updated_at": cache["updated_at"],
        "items": items[offset:offset + page_size],
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "has_previous": page > 1,
        "has_next": page < total_pages,
    }


app.mount("/", StaticFiles(directory=Path(__file__).resolve().parents[1] / "frontend",
                           html=True), name="frontend")
