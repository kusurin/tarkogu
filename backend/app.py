# backend/app.py
import asyncio
import json
import os
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from tarkogu.get_data import get_item_metadata, refresh_item_historical_price_cache, UPDATE_INTERVAL
from tarkogu.predict import predict_price_and_offer_count, find_buy_times, find_sell_times
from tarkogu.utils import get_pkg_abs_path

PRED_CATEGORIES = {'Barter item', 'Item', 'Food and drink', 'Key', 'Equipment'}
REF_DAYS = 14
PREDICT_HOURS = 24

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本地测试先放开，部署后改成你的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_PATH = Path(get_pkg_abs_path()) / "data" / "rankings_cache.json"

W_SORT = {"buy_range": 0.5, "sell_range": 0, "rmse": -2.0}

DEFAULT_CACHE = {"updated_at": None, "items": []}
NUMERIC_FIELDS = {"score", "buy_range", "sell_range", "rmse", "offer_yhat_mean_24h"}

def save_cache(data: dict) -> None:
    tmp_path = CACHE_PATH.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp_path, CACHE_PATH)

def load_cache() -> dict:
    if not CACHE_PATH.exists():
        return DEFAULT_CACHE.copy()
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return DEFAULT_CACHE.copy()
    if not isinstance(data, dict):
        return DEFAULT_CACHE.copy()
    items = data.get("items", [])
    if isinstance(items, list):
        items = sanitize_items(items)
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

async def update_loop():
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

            price_by_id = refresh_item_historical_price_cache(items, REF_DAYS)

            cache = load_cache()
            items_by_id = build_items_by_id(cache.get("items", []))

            for item_id in items:
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

                    correct_factor = 1 / res["price_yhat"].mean()

                    score = (
                        W_SORT["buy_range"] * buy_range * correct_factor +
                        W_SORT["sell_range"] * sell_range * correct_factor +
                        W_SORT["rmse"] * rmse * correct_factor
                    )

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
                    print(price_df.columns) 
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
        finally:
            if sleep_seconds is None:
                elapsed = time.monotonic() - cycle_start
                sleep_seconds = max(0, UPDATE_INTERVAL / 1000 - elapsed)
            if sleep_seconds > 0:
                await asyncio.sleep(sleep_seconds)

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(update_loop())

@app.get("/rankings")
def get_rankings():
    return load_cache()