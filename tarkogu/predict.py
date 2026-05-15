from tarkogu.get_data import fetch_item_historical_price
import pandas as pd
from prophet import Prophet

def predict_price_and_offer_count(id, ref_days, predict_hours,  return_details=False):
    price_df = fetch_item_historical_price(id, ref_days)

    price_df.rename(columns={'time': 'ds', 'price': 'y'}, inplace=True)

    offer_df = price_df[['ds', 'offerCount']].copy()
    offer_df.rename(columns={'offerCount': 'y'}, inplace=True)

    # Predict price
    price_model = Prophet(changepoint_prior_scale=0.01)
    price_model.fit(price_df[['ds', 'y']])
    price_future = price_model.make_future_dataframe(periods=predict_hours, freq='h')
    price_forecast = price_model.predict(price_future)

    # Predict offer count
    offer_model = Prophet(changepoint_prior_scale=0.01)
    offer_model.fit(offer_df[['ds', 'y']])
    offer_future = offer_model.make_future_dataframe(periods=predict_hours, freq='h')
    offer_forecast = offer_model.predict(offer_future)

    if return_details:
        return price_forecast, offer_forecast, price_model, offer_model
    
    price_forecast.columns = ['price_' + _col for _col in price_forecast.columns]
    offer_forecast.columns = ['offer_' + _col for _col in offer_forecast.columns]

    result_df = pd.concat([price_forecast, offer_forecast], axis=1)

    return result_df