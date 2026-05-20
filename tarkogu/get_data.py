import requests
import pandas as pd
import os

from tarkogu.utils import get_pkg_abs_path

INT_MAX = 2**31 - 1

API_URL = "https://api.tarkov.dev/graphql"
UPDATE_INTERVAL = 1000 * 60 * 60 * 2 # 2 hours
MS_IN_DAY = 1000 * 60 * 60 * 24
CURRENT_DIR = get_pkg_abs_path()

def get_item_metadata():
    if os.path.exists(CURRENT_DIR / "data" / "items_metadata.csv"):
        return pd.read_csv(CURRENT_DIR / "data" / "items_metadata.csv", index_col=0, header=0)
    query = """
    query {
        items(lang: en) {
            id
            name
            iconLink
            category {
                name
                parent {
                    name
                }
            }
        }
    }"""

    response = requests.post(API_URL, json={'query': query})
    data = response.json()

    meta_df = pd.DataFrame(data['data']['items'])
    meta_df['category'] = meta_df['category'].apply(lambda x: x['parent']['name'])

    # zh
    query = """
    query {
        items(lang: zh) {
            id
            name
        }
    }"""

    response = requests.post(API_URL, json={'query': query})
    data = response.json()

    meta_df['name_zh'] = meta_df['id'].map({item['id']: item['name'] for item in data['data']['items']})

    meta_df.to_csv('data/items_metadata.csv', index=False, header=True)
    return meta_df


def get_item_historical_price(id, days, limit = INT_MAX, offset = 0):
    query = """
    query ($id: ID!, $days: Int!, $limit: Int!, $offset: Int!) {
        historicalItemPrices(id: $id, days: $days, limit: $limit, offset: $offset) {
            price
            priceMin
            offerCount
            timestamp
        }
    }"""

    response = requests.post(API_URL, json={'query': query, 'variables': {'id': id, 'days': int(days), 'limit': int(limit), 'offset': int(offset)}})
    data = response.json()

    price_df = pd.DataFrame(data['data']['historicalItemPrices'])

    if not price_df.empty:
        price_df['timestamp'] = price_df['timestamp'].astype('int64')
        price_df['offerCount'] = price_df['offerCount'].astype('int64')
        price_df['time'] = pd.to_datetime(price_df['timestamp'], unit='ms')

    return price_df

def _build_historical_price_batch_query(item_ids, days, limit, offset):
    alias_map = {}
    lines = ["query {"]
    for idx, item_id in enumerate(item_ids):
        alias = f"item_{idx}"
        alias_map[alias] = item_id
        lines.append(
            f'  {alias}: historicalItemPrices(id: "{item_id}", days: {int(days)}, limit: {int(limit)}, offset: {int(offset)}) {{'
        )
        lines.append("    price")
        lines.append("    priceMin")
        lines.append("    offerCount")
        lines.append("    timestamp")
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines), alias_map

def _normalize_price_df(price_df):
    df = price_df.copy()
    for col in ["timestamp", "price", "priceMin", "offerCount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "timestamp" in df.columns:
        df = df.dropna(subset=["timestamp"])
        df["timestamp"] = df["timestamp"].astype("int64")

    if "time" not in df.columns or df["time"].isna().all():
        if "timestamp" in df.columns:
            df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    else:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_convert(None)

    return df

def _load_price_cache(item_id):
    path = CURRENT_DIR / "data" / f"{item_id}_historical_price.csv"
    if not os.path.exists(path):
        return None
    try:
        price_df = pd.read_csv(path, index_col=False, header=0)
    except (OSError, pd.errors.ParserError):
        return None
    if price_df.empty:
        return price_df
    return _normalize_price_df(price_df)

def _save_price_cache(item_id, price_df):
    path = CURRENT_DIR / "data" / f"{item_id}_historical_price.csv"
    price_df.to_csv(path, index=False, header=True)

def _needs_refresh(price_df, current_timestamp, start_timestamp):
    if price_df is None or price_df.empty:
        return True
    if "timestamp" not in price_df.columns:
        return True

    price_df = price_df.dropna(subset=["timestamp"])
    if price_df.empty:
        return True

    last_ts = price_df["timestamp"].max()
    first_ts = price_df["timestamp"].min()

    if last_ts < current_timestamp - UPDATE_INTERVAL:
        return True

    if first_ts > start_timestamp:
        return True

    timestamp_diffs = price_df["timestamp"].sort_values().diff()
    if (timestamp_diffs >= UPDATE_INTERVAL * 7).any():
        return True

    return False

def get_items_historical_price_batch(item_ids, days, limit=INT_MAX, offset=0):
    if not item_ids:
        return {}

    query, alias_map = _build_historical_price_batch_query(item_ids, days, limit, offset)
    response = requests.post(API_URL, json={"query": query})
    data = response.json()

    if "errors" in data:
        print(data["errors"])

    results = {}
    payload = data.get("data", {})

    for alias, item_id in alias_map.items():
        records = payload.get(alias, [])
        price_df = pd.DataFrame(records)
        if not price_df.empty:
            price_df = _normalize_price_df(price_df)
        results[item_id] = price_df

    return results

def refresh_item_historical_price_cache(item_ids, ref_days):
    current_timestamp = pd.Timestamp.now(tz="UTC").timestamp() * 1000
    start_timestamp = current_timestamp - ref_days * MS_IN_DAY - UPDATE_INTERVAL

    cached_by_id = {}
    ids_to_refresh = []

    for item_id in item_ids:
        cached_df = _load_price_cache(item_id)
        cached_by_id[item_id] = cached_df
        if _needs_refresh(cached_df, current_timestamp, start_timestamp):
            ids_to_refresh.append(item_id)

    if ids_to_refresh:
        fetched_by_id = get_items_historical_price_batch(ids_to_refresh, ref_days)
        for item_id in ids_to_refresh:
            fetched_df = fetched_by_id.get(item_id)
            if fetched_df is None or fetched_df.empty:
                continue
            cached_df = cached_by_id.get(item_id)
            if cached_df is None or cached_df.empty:
                merged_df = fetched_df
            else:
                merged_df = pd.concat([cached_df, fetched_df], ignore_index=True)
                merged_df.drop_duplicates("timestamp", inplace=True)
                merged_df.sort_values(by="timestamp", inplace=True)
            merged_df = _normalize_price_df(merged_df)
            _save_price_cache(item_id, merged_df)
            cached_by_id[item_id] = merged_df

    filtered_by_id = {}
    for item_id, price_df in cached_by_id.items():
        if price_df is None or price_df.empty:
            continue
        price_df = _normalize_price_df(price_df)
        price_df = price_df[price_df["timestamp"] >= start_timestamp].copy()
        filtered_by_id[item_id] = price_df

    return filtered_by_id

def fetch_item_historical_price(id, days):
    current_timestamp = pd.Timestamp.now(tz='UTC').timestamp() * 1000
    
    start_timestamp = current_timestamp - days * MS_IN_DAY - UPDATE_INTERVAL

    if not os.path.exists(CURRENT_DIR / "data" / f"{id}_historical_price.csv"):
        price_df = get_item_historical_price(id, days)
    else:
        while True:
            price_df = pd.read_csv(CURRENT_DIR / "data" / f"{id}_historical_price.csv", index_col=False, header=0)

            price_df_available = price_df[price_df['timestamp'] >= start_timestamp].copy()

            if price_df_available.empty:
                append_price_df = get_item_historical_price(id, days)
                break

            price_df_available_pad_start = pd.DataFrame([{
                'timestamp': start_timestamp,
                'price': price_df_available.iloc[0,:]['price'],
                'priceMin': price_df_available.iloc[0,:]['priceMin'],
                'offerCount': price_df_available.iloc[0,:]['offerCount'],
                'time': pd.to_datetime(price_df_available.iloc[0,:]['timestamp'] - UPDATE_INTERVAL, unit='ms')
            }])

            price_df_available_pad_end = pd.DataFrame([{
                'timestamp': current_timestamp,
                'price': price_df_available.iloc[-1,:]['price'],
                'priceMin': price_df_available.iloc[-1,:]['priceMin'],
                'offerCount': price_df_available.iloc[-1,:]['offerCount'],
                'time': pd.to_datetime(price_df_available.iloc[-1,:]['timestamp'] + UPDATE_INTERVAL, unit='ms')
            }])

            is_outdated = (current_timestamp - price_df_available.iloc[-1]['timestamp'] >= UPDATE_INTERVAL)

            price_df_available = pd.concat([price_df_available_pad_start, price_df_available, price_df_available_pad_end], ignore_index=True)

            price_df_available['timestamp_diffs'] = price_df_available['timestamp'].diff()
            price_df_available['should_fetch'] = price_df_available['timestamp_diffs'] >= UPDATE_INTERVAL * 7

            append_price_df = pd.DataFrame()

            if price_df_available['should_fetch'].any():
                
                idx_should_fetch_first = price_df_available[price_df_available['should_fetch']].index[0]
                idx_should_fetch_last = price_df_available[price_df_available['should_fetch']].index[-1]

                if is_outdated:
                    idx_should_fetch_last = price_df_available.index[-1]

                fetch_time_start = price_df_available.iloc[idx_should_fetch_first - 1,:]['timestamp']
                fetch_time_end = price_df_available.iloc[min(idx_should_fetch_last + 1, len(price_df_available) - 1),:]['timestamp']

            elif is_outdated:
                fetch_time_start = price_df_available.iloc[-2,:]['timestamp']
                fetch_time_end = price_df_available.iloc[-1,:]['timestamp']
            
            else:
                print('Using cache')
                break

            # 不包含start和end
            record_limit = (fetch_time_end - fetch_time_start) // (UPDATE_INTERVAL) + 1

            offset_days = (current_timestamp - fetch_time_start) // (1000 * 60 * 60 * 24) + 1
            offset_interval = (current_timestamp - fetch_time_start) // (UPDATE_INTERVAL) + 1

            request_offset = offset_days * (MS_IN_DAY / UPDATE_INTERVAL) - offset_interval

            append_price_df = get_item_historical_price(id, offset_days, limit=record_limit, offset=request_offset)
                
            price_df = pd.concat([price_df, append_price_df], ignore_index=True)

            price_df.drop_duplicates('timestamp', inplace=True)
            price_df.sort_values(by='timestamp', inplace=True)

            break
    
    price_df['time'] = pd.to_datetime(price_df['time'], utc=True).dt.tz_convert(None)

    price_df.to_csv(CURRENT_DIR / "data" / f"{id}_historical_price.csv", index=False, header=True)

    return price_df[price_df['timestamp'] >= (current_timestamp - days * MS_IN_DAY - UPDATE_INTERVAL)].copy()