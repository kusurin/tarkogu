from tarkogu.get_data import fetch_item_historical_price
import pandas as pd
import numpy as np
from prophet import Prophet

def predict_price_and_offer_count(id, ref_days, predict_hours,  return_details=False, price_df=None):
    if price_df is None:
        price_df = fetch_item_historical_price(id, ref_days)

    if price_df.empty:
        raise ValueError(f"No price data available for {id}")

    if "time" not in price_df.columns and "timestamp" in price_df.columns:
        price_df["time"] = pd.to_datetime(price_df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)

    price_df.rename(columns={'time': 'ds', 'priceMin': 'y'}, inplace=True)

    offer_df = price_df[['ds', 'offerCount']].copy()
    offer_df.rename(columns={'offerCount': 'y'}, inplace=True)

    offer_model = Prophet()
    offer_model.fit(offer_df[['ds', 'y']])
    offer_future = offer_model.make_future_dataframe(periods=predict_hours, freq='2h')
    offer_forecast = offer_model.predict(offer_future)

    # Predict price
    price_model = Prophet()
    price_model.add_regressor('offerCount')
    price_model.fit(price_df[['ds', 'y', 'offerCount']])
    price_future = price_model.make_future_dataframe(periods=predict_hours, freq='2h')
    offer_regressor = offer_forecast[['ds', 'yhat']].rename(columns={'yhat': 'offerCount'})
    price_future = price_future.merge(offer_regressor, on='ds', how='left')

    if price_future['offerCount'].isnull().any():
        fallback = price_df[['ds', 'offerCount']]
        price_future = price_future.merge(fallback, on='ds', how='left', suffixes=('', '_hist'))
        price_future['offerCount'] = price_future['offerCount'].fillna(price_future['offerCount_hist'])
        price_future.drop(columns=['offerCount_hist'], inplace=True)

    price_forecast = price_model.predict(price_future)

    if return_details:
        return price_forecast, offer_forecast, price_model, offer_model
    
    price_forecast.columns = ['price_' + _col for _col in price_forecast.columns]
    offer_forecast.columns = ['offer_' + _col for _col in offer_forecast.columns]

    result_df = pd.concat([price_forecast, offer_forecast], axis=1)

    result = pd.merge(price_df, result_df, left_on='ds', right_on='price_ds', how='inner')
    result = result.dropna(subset=['y', 'price_yhat'])
    mse = np.mean((result['y'] - result['price_yhat']) ** 2)
    rmse = np.sqrt(mse)

    return result_df, rmse

def predict_price_and_offer_count_test(id, ref_days, predict_hours,  return_details=False, train_fraction=0.8):
    price_df = fetch_item_historical_price(id, ref_days)

    price_df.rename(columns={'time': 'ds', 'priceMin': 'y'}, inplace=True)

    offer_df = price_df[['ds', 'offerCount', 'timestamp']].copy()
    offer_df.rename(columns={'offerCount': 'y'}, inplace=True)

    train_test_split_time = price_df['timestamp'].min() + (price_df['timestamp'].max() - price_df['timestamp'].min()) * train_fraction
    
    price_df_train = price_df[price_df['timestamp'] < train_test_split_time].copy()
    price_df_test = price_df[price_df['timestamp'] >= train_test_split_time].copy()

    offer_df_train = offer_df[offer_df['timestamp'] < train_test_split_time].copy()
    offer_df_test = offer_df[offer_df['timestamp'] >= train_test_split_time].copy()

    offer_model = Prophet()
    offer_model.fit(offer_df_train[['ds', 'y']])
    offer_future = offer_model.make_future_dataframe(periods=predict_hours, freq='2h')
    offer_forecast = offer_model.predict(offer_future)

    # Predict price
    price_model = Prophet()
    price_model.add_regressor('offerCount')
    price_model.fit(price_df_train[['ds', 'y', 'offerCount']])
    price_future = price_model.make_future_dataframe(periods=predict_hours, freq='2h')
    offer_regressor = offer_forecast[['ds', 'yhat']].rename(columns={'yhat': 'offerCount'})
    price_future = price_future.merge(offer_regressor, on='ds', how='left')

    if price_future['offerCount'].isnull().any():
        fallback = price_df_train[['ds', 'offerCount']]
        price_future = price_future.merge(fallback, on='ds', how='left', suffixes=('', '_hist'))
        price_future['offerCount'] = price_future['offerCount'].fillna(price_future['offerCount_hist'])
        price_future.drop(columns=['offerCount_hist'], inplace=True)

    price_forecast = price_model.predict(price_future)

    if return_details:
        return price_forecast, offer_forecast, price_model, offer_model, price_df_train, price_df_test, offer_df_train, offer_df_test
    
    price_forecast.columns = ['price_' + _col for _col in price_forecast.columns]
    offer_forecast.columns = ['offer_' + _col for _col in offer_forecast.columns]

    result_df = pd.concat([price_forecast, offer_forecast], axis=1)

    return result_df

# 当时价格和预测价格中，后续价格大于其超过10000的时间点买入
def find_buy_times(price_forecast):
    current_time = pd.Timestamp.now(tz='UTC').tz_localize(None)
    last_idx = price_forecast[price_forecast['price_ds'] <= current_time].index[-1]

    price_forecast = price_forecast.iloc[last_idx:][::-1].copy()
    price_forecast.reset_index(drop=True, inplace=True)

    max_follow = float('-inf')
    
    price_forecast['max_follow_diff'] = 0.0

    for idx, row in price_forecast.iterrows():
        price_forecast.loc[idx, 'max_follow_diff'] = max_follow - row['price_yhat']
        max_follow = max(max_follow, row['price_yhat'])
    
    price_forecast['should_buy'] = price_forecast['max_follow_diff'] >= 10000

    buy_range = price_forecast['max_follow_diff'].max()

    return price_forecast[price_forecast['should_buy']]['price_ds'].tolist(), buy_range


# 当时价格和预测价格中，前24h内有价格小于其超过10000的时间点卖出
def find_sell_times(price_forecast, previous_ref_hours=24):
    current_time = pd.Timestamp.now(tz='UTC').tz_localize(None)

    window_start = current_time - pd.Timedelta(hours=previous_ref_hours)
    price_forecast = price_forecast[price_forecast['price_ds'] >= window_start].copy()
    price_forecast.reset_index(drop=True, inplace=True)

    if price_forecast.empty:
        return []

    last_idx = price_forecast[price_forecast['price_ds'] <= current_time].index[-1]
    if last_idx <= 0:
        return []

    min_previous = price_forecast.loc[:last_idx, 'price_yhat'].min()
    price_forecast['max_previous_diff'] = 0.0

    for idx in range(last_idx, len(price_forecast)):
        price_forecast.loc[idx, 'max_previous_diff'] = price_forecast.loc[idx, 'price_yhat'] - min_previous
        min_previous = min(min_previous, price_forecast.loc[idx, 'price_yhat'])

    price_forecast['should_sell'] = price_forecast['max_previous_diff'] >= 10000

    sell_range = price_forecast['max_previous_diff'].max()

    return price_forecast[price_forecast['should_sell']]['price_ds'].tolist(), sell_range