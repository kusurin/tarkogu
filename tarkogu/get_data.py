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
    if os.path.exists(CURRENT_DIR / "data" / "item_metadata.csv"):
        return pd.read_csv(CURRENT_DIR / "data" / "item_metadata.csv", index_col=0, header=0)
    query = """
    query {
        items(lang: en) {
            id
            name
        }
    }"""

    response = requests.post(API_URL, json={'query': query})
    data = response.json()

    meta_df = pd.DataFrame(data['data']['items'])
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

    price_df['timestamp'] = price_df['timestamp'].astype('int64')
    price_df['offerCount'] = price_df['offerCount'].astype('int64')
    price_df['time'] = pd.to_datetime(price_df['timestamp'], unit='ms')

    return price_df

def fetch_item_historical_price(id, days):
    current_timestamp = pd.Timestamp.now(tz='UTC').timestamp() * 1000

    if not os.path.exists(CURRENT_DIR / "data" / f"{id}_historical_price.csv"):
        price_df = get_item_historical_price(id, days)
    else:
        price_df = pd.read_csv(CURRENT_DIR / "data" / f"{id}_historical_price.csv", index_col=False, header=0)
    
        price_df_available = price_df[price_df['timestamp'] >= (pd.Timestamp.now().timestamp() * 1000 - days * MS_IN_DAY - UPDATE_INTERVAL)].copy()
        if price_df_available.empty:
            append_price_df = get_item_historical_price(id, days)

        else:
            price_df_available['timestamp_diffs'] = price_df_available['timestamp'].diff()
            price_df_available['should_fetch'] = price_df_available['timestamp_diffs'] >= UPDATE_INTERVAL * 2

            is_outdated = current_timestamp - price_df_available.iloc[-1,:]['timestamp'] >= UPDATE_INTERVAL * 2

            append_price_df = pd.DataFrame()

            if is_outdated or price_df_available['should_fetch'].any():
                
                idx_should_fetch_first = price_df_available[price_df_available['should_fetch']].index[0]
                idx_should_fetch_last = price_df_available[price_df_available['should_fetch']].index[-1]

                fetch_time_start = price_df_available.iloc[idx_should_fetch_first - 1,:]['timestamp']
                fetch_time_end = price_df_available.iloc[idx_should_fetch_last + 1,:]['timestamp']

                if price_df_available.iloc[0,:]['timestamp'] - (current_timestamp - days * MS_IN_DAY) >= UPDATE_INTERVAL * 2:
                    fetch_time_start = current_timestamp - days * MS_IN_DAY

                if is_outdated:
                    fetch_time_end = current_timestamp

                # 不包含start和end
                record_limit = (fetch_time_end - fetch_time_start) // (UPDATE_INTERVAL) - 1

                offset_days = (current_timestamp - fetch_time_start) // (1000 * 60 * 60 * 24) + 1
                offset_interval = (current_timestamp - fetch_time_start) // (UPDATE_INTERVAL) + 1

                request_offset = offset_days * (MS_IN_DAY / UPDATE_INTERVAL) - offset_interval

                append_price_df = get_item_historical_price(id, offset_days, limit=record_limit, offset=request_offset)

        price_df = pd.concat([price_df, append_price_df], ignore_index=True)

        price_df.drop_duplicates('timestamp', inplace=True)
        price_df.sort_values(by='timestamp', inplace=True)

    price_df.to_csv(CURRENT_DIR / "data" / f"{id}_historical_price.csv", index=False, header=True)

    return price_df[price_df['timestamp'] >= (current_timestamp - days * MS_IN_DAY - UPDATE_INTERVAL)].copy()