import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
import matplotlib.pyplot as plt
import os

# To ignore warnings from ARIMA
warnings.filterwarnings("ignore")

class ARIMAModel:
    def __init__(self, data, p, d, q):

        self.p = p
        self.d = d
        self.q = q
        self.data = data
        self.model = None
        self.fitted_model = None
        self.steps = None

    def load_data(self):
        # TODO: ADF Test
        # adf_result = self.perform_adf_test(self.data)
        
        # # # Check if differencing is needed
        # if adf_result[1] > 0.05:  # Non-stationary
        self.data['Store_Number'] =  self.data['Store'].str.split('_').str[1]
        self.data['Department_Number'] =  self.data['DEPARTMENT'].str.extract(r'(\d+)')
        self.data['SKU_Number'] =   self.data['SKU'].str.extract(r'(\d+)')
        self.data['Category_Number'] =  self.data['CATEGORY'].str.extract(r'(\d+)')
        self.data['Week'] = pd.to_datetime( self.data['Week'])
    
    def perform_adf_test(self, data):
        adf_result = adfuller(data['Units_Sold'])
        return adf_result
    
    def fit(self, sku_data):
        self.model = ARIMA(sku_data['Units_Sold'], order=(self.p, self.d, self.q))
        self.fitted_model = self.model.fit()
    
    def predict(self):
        if self.fitted_model is None:
            raise ValueError("The model has not been fitted yet.")
        self.store_forecasts = self.fitted_model.forecast(steps=self.steps)
        
    
    def fit_sku(self):
        store_forecasts = {}
        unique_skus = self.data['SKU_Number'].unique()
        for sku in unique_skus:
            sku_data = self.data[self.data['SKU_Number'] == sku].copy()
            sku_data.set_index('Week', inplace=True)
            sku_data  = sku_data.resample('W').sum()  # Weekly data
            if len(sku_data) < 10:
                continue
            try:
                self.fit(sku_data)
                self.predict()
                store_forecasts[sku] = self.store_forecasts
            except Exception as e:
                print(f"Could not fit model for SKU {sku} due to: {e}")
        return store_forecasts
    def summary(self):

        if self.fitted_model is None:
            raise ValueError("The model has not been fitted yet.")
        return self.fitted_model.summary()
    
    def create_forecast_dataframe(self, store_name, forecasts):
        forecast_list = []
        
        for sku, forecast in forecasts.items():
            for week_offset in range(len(forecast)):
                forecast_list.append({
                    'Store_Name': store_name,
                    'SKU_Number': sku,
                    'Forecast_Week': pd.Timestamp.now() + pd.DateOffset(weeks=week_offset + 1),
                    'Forecasted_Units_Sold': forecast[week_offset]
                })
        
        return pd.DataFrame(forecast_list)
    
    def run_forecasting(self):
        # store_names = self.data['Store_Number'].unique()
        combined_forecasts = []
        stores =['6']
        self.steps = 7
        for store in stores:
            self.load_data()
            self.data = self.data[self.data['Store_Number'] == store].copy()
            forecasts = self.fit_sku()
            forecast_df = self.create_forecast_dataframe(store, forecasts)
            combined_forecasts.append(forecast_df)

        if combined_forecasts:
            all_forecasts_df = pd.concat(combined_forecasts, ignore_index=True)
            return all_forecasts_df
        return pd.DataFrame()  # Return an empty DataFrame if no forecasts
        # return forecasts
    def create_forecast_dataframe(self, store_name, forecasts):
        forecast_list = []
        
        for sku, forecast in forecasts.items():
            for week_offset in range(len(forecast)):
                forecast_list.append({
                    'Store_Name': store_name,
                    'SKU_Number': sku,
                    'Forecast_Week': pd.Timestamp.now() + pd.DateOffset(weeks=week_offset + 1),
                    'Forecasted_Units_Sale': forecast[week_offset]
                })
        
        return pd.DataFrame(forecast_list)
def main():
    file_path = ''
    input_file_path = os.path.join(os.path.dirname(file_path), 'Capstone_Dataset.csv')
    data = pd.read_csv(input_file_path)
    arima = ARIMAModel(data, p=1, d=1, q=1)
    store_forecasts = arima.run_forecasting()
    output_file_path = os.path.join(os.path.dirname(file_path), 'forecasted_sales.csv')
    store_forecasts.to_csv(output_file_path, index=False) 

if __name__ == "__main__":
    main()


        
    
