import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import holidays
from statsmodels.tsa.statespace.sarimax import SARIMAX
import yfinance as yf
from datetime import datetime
import pandas_datareader as pdr

# To ignore warnings from ARIMA
warnings.filterwarnings("ignore")

class ARIMAModel:
    def __init__(self, data, p, d, q):

        self.p = p
        self.d = d
        self.q = q
        self.data = None
        self.origin_data = data
        self.index_data = None
        self.model = None
        self.fitted_model = None
        self.steps = None
        self.accuracy_metrics = {}

    def load_data(self):
        # TODO: ADF Test
        # adf_result = self.perform_adf_test(self.data)
        
        # # # Check if differencing is needed
        # if adf_result[1] > 0.05:  # Non-stationary
        years = [2020, 2021, 2023, 2024, 2025]  # Add more years as needed
        self.origin_data['Store_Number'] =  self.origin_data['Store'].str.split('_').str[1]
        self.origin_data['Department_Number'] =  self.origin_data['DEPARTMENT'].str.extract(r'(\d+)')
        self.origin_data['SKU_Number'] =   self.origin_data['SKU'].str.extract(r'(\d+)')
        self.origin_data['Category_Number'] =  self.origin_data['CATEGORY'].str.extract(r'(\d+)')
        self.origin_data['Week'] = pd.to_datetime( self.origin_data['Week']).dt.tz_localize(None)
        us_holidays = holidays.CountryHoliday('US', years=years)
        holiday_dates = pd.to_datetime(list(us_holidays.keys()))
        self.origin_data['is_week_before_holiday'] = self.origin_data['Week'].apply(
            lambda x: int(any((0 < (holiday - x).days <= 7) for holiday in holiday_dates))
        )
        self.origin_data['is_week_before_holiday_lag1'] = self.origin_data['is_week_before_holiday'].shift(1)
        self.origin_data['is_week_before_holiday_lag7'] = self.origin_data['is_week_before_holiday'].shift(7)


        start = datetime(2011, 1, 1)
        end = datetime.now()

        # Load CPI Index
        cpi_data = yf.download('CPI', start, end)  
        cpi_data.reset_index(inplace=True)
        cpi_data.head()
        # Flattening MultiIndex columns
        cpi_data.columns = ['_'.join(col).strip() for col in cpi_data.columns.values]
        # Renaming flattened columns
        cpi_data.rename(columns={
            'Date_': 'Date',
            'Adj Close_CPI': 'Adj_Close',
            'Close_CPI': 'Close_CPI',
            'High_CPI': 'High_CPI',
            'Low_CPI': 'Low_CPI',
            'Open_CPI': 'Open_CPI',
            'Volume_CPI': 'Volume_CPI'
        }, inplace=True)
        # Merge the target data with CPI data
        cpi_data['Date'] = pd.to_datetime(cpi_data['Date']).dt.tz_localize(None)
        # Load GDP Index
        gdp_data = pdr.get_data_fred('GDP', start=start, end=end)
        gdp_data.reset_index(inplace=True)
        gdp_data['DATE'] = pd.to_datetime(gdp_data['DATE']).dt.tz_localize(None)
        self.index_data = pd.merge(cpi_data, gdp_data, left_on='Date', right_on='DATE', how='outer')
        self.index_data['gdp_lag1'] = self.index_data['GDP'].shift(1)
        self.index_data['gdp_lag3'] = self.index_data['GDP'].shift(3)
        self.index_data['cpi_lag1'] = self.index_data['Adj_Close'].shift(1)
        self.index_data['cpi_lag3'] = self.index_data['Adj_Close'].shift(3)
        # merge original dataset with index dataset
        self.data = pd.merge(self.origin_data , self.index_data, left_on='Week', right_on='Date', how='left')


    def perform_adf_test(self, data):
        adf_result = adfuller(data['Units_Sold'])
        return adf_result
    
    def fit(self, y, x):
        # self.model = ARIMA(y, order=(self.p, self.d, self.q))
        self.model = SARIMAX(y, exog=x, order=(1, 1, 1))  # Example parameters
        self.fitted_model = self.model.fit()
    
    def predict(self, y, y_test, x_test):
        if self.fitted_model is None:
            raise ValueError("The model has not been fitted yet.")
        # self.store_forecasts = self.fitted_model.forecast(steps=self.steps)
        y_pred = self.fitted_model.predict(start=len(y), end=len(y) + len(y_test) - 1, exog=x_test)
        return y_pred
    
    def fit_sku(self, store):
        store_forecasts = {}
        store_prediction = {}
        store_accuracy = []
        count = 0
        unique_skus = self.data['SKU_Number'].unique()
        for sku in unique_skus:
            sku_data = self.data[self.data['SKU_Number'] == sku].copy()
            sku_data['Week'] = pd.to_datetime(sku_data['Week'], errors='coerce').dt.tz_localize(None)
            sku_data.set_index('Week', inplace=True)
            # Drop non-numeric columns (if any) before resampling
            numeric_sku_data = sku_data.select_dtypes(include='number')
            sku_data  = numeric_sku_data.resample('W').sum()  # Weekly data
            # Split model
            train_size = int(len(sku_data) * 0.8)
            y = sku_data['Units_Sold']
            
            X = sku_data[['is_week_before_holiday', 'is_week_before_holiday_lag1', 'is_week_before_holiday_lag7', 'gdp_lag1', 'gdp_lag3','cpi_lag1', 'cpi_lag3' ]]
            y_train, y_test = y[:train_size], y[train_size:]
            X_train, X_test = X[:train_size], X[train_size:]
            if len(y_train) < 10:
                continue
            try:
                # Get actual values for the next 7 weeks
                self.fit(y_train, X_train)
                y_pred = self.predict(y_train, y_test, X_test)

                # Evaluate the model
                mse = mean_squared_error(y_test, y_pred)
                # mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                print(f'Mean Squared Error: {mse}')
                # Store the results
                store_accuracy.append({
                    'store_id': store,
                    'sku_id': sku,
                    'mse': mse,
                    'rmse': rmse,
                })
                # store prediction               

                store_prediction[sku] = y_pred
                # store_forecasts[sku] = self.store_forecasts
            except Exception as e:
                count= count + 1
                print(f"Could not fit model for SKU {sku} due to: {e}", count)
        return pd.DataFrame(store_accuracy), store_prediction
        # return store_forecasts, store_accuracy
    def summary(self):

        if self.fitted_model is None:
            raise ValueError("The model has not been fitted yet.")
        return self.fitted_model.summary()
    
    def create_prediction_dataframe(self, store_number, prediction):
        prediction_list = []

        for sku, prediction in prediction.items():
            for week_offset in range(len(prediction)):
                prediction_list.append({
                    'Store_Number': store_number,
                    'SKU_Number': sku,
                    'Forecast_Week': pd.Timestamp.now() + pd.DateOffset(weeks=week_offset + 1),
                    'Forecasted_Units_Sold': prediction[week_offset]
                })
        
        return pd.DataFrame(prediction_list)
    
    def run_forecasting(self, data):
        self.load_data()
        stores = data['Store_Number'].unique()
        combined_forecasts = []
        combined_metrics = []
        combined_prediction = []
        stores = ['6']
        self.steps = 7
        for store in stores:
            # reload the data at each store
            self.original_data = data
            self.load_data()
            self.data = self.data[self.data['Store_Number'] == store].copy()
            # forecasts, store_accuracy = self.fit_sku(store)
            store_metrics, store_prediction = self.fit_sku(store)
            prediction_df = self.create_prediction_dataframe(store, store_prediction)

            # forecast_df = self.create_forecast_dataframe(store, forecasts)
            combined_metrics.append(store_metrics)
            combined_prediction.append(prediction_df)

        # if combined_prediction:
        #     all_prediction_df = pd.concat(combined_forecasts, ignore_index=True)
        #     all_metric_df = pd.concat(combined_metrics, ignore_index=True)
        return prediction_df, store_metrics
        return pd.DataFrame()  # Return an empty DataFrame if no forecasts

def main():
    file_path = '/Users/anabellaisaro/Documents/Documents - Anabella’s MacBook Pro/Northwestern/Projects/Deloitte/forecast/data/'
    input_file_path = os.path.join(os.path.dirname(file_path), 'Capstone_Dataset.csv')
    data = pd.read_csv(input_file_path)
    arima = ARIMAModel(data, p=1, d=1, q=1)
    all_prediction_df, all_metric_df = arima.run_forecasting(data)
    output_file_path = os.path.join(os.path.dirname(file_path), 'forecasted_sales.csv')
    pred_output_file_path = os.path.join(os.path.dirname(file_path), 'prediction.csv')
    metric_output_file_path = os.path.join(os.path.dirname(file_path), 'metrics.csv')

    all_prediction_df.to_csv(pred_output_file_path, index=False) 
    all_metric_df.to_csv(metric_output_file_path, index=False) 

if __name__ == "__main__":
    main()


        
    
