import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import holidays
import yfinance as yf
import pandas_datareader as pdr
from datetime import datetime
import os
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")


class ExponentialSmoothingModel:
    def __init__(self, data):
        self.origin_data = data
        self.index_data = None
        self.data = data
        self.steps = None

    def load_data(self):
        self.data['Store_Location'] =  self.data['Store_Num'].str.split('_').str[1]
        self.data['Store_Location'] =  self.data['Store_Location'].astype(int, errors='ignore')  # Convert to int, ignore errors for non-convertible values
        self.data['Store_Number'] =  self.data['Store_Num'].str.extract(r'(\d+)')
        self.data['Store_Number'] =  self.data['Store_Number'].astype(int, errors='ignore')  # Convert to int
        self.data['SKU_Number'] =  self.data['SKU'].str.extract(r'(\d+)')
        self.data['SKU_Number'] =  self.data['SKU_Number'].astype(int, errors='ignore')  # Convert to int
        self.data['Week'] = pd.to_datetime( self.data['Week'])
        self.data =  self.data.rename(columns={"Units_Sold": "UNITS_SOLD"})

    def fit_forecast(self):
        group_accuracy = []
        group_prediction = {}
        store_data = self.data

        # Filter for specific store, SKU, and Category
        group_data = store_data[
            (store_data['Store_Location'] == 'Dallas') &
            (store_data['Category'] == 'Shoes') &
            # (store_data['Department'] == 'Grocery') &
            (store_data['SKU_Number'] == 326) 
        ]

        group_data.set_index('Week', inplace=True)
        group_data = group_data.resample('W').sum()
        start_week='2023-07-01'
        y = group_data['UNITS_SOLD']
        start_index = group_data.index[group_data.index >= pd.Timestamp(start_week)]
        if start_index.empty:
            print(f"No data available starting from {start_week}.")
            return pd.DataFrame(group_accuracy), group_prediction
        train_size = int(len(group_data) * 0.8)
        y = group_data['UNITS_SOLD']

        y_train, y_test = y[:train_size], y[train_size:]

        if len(y_train) < 10:
            print("y_train has fewer than 10 elements. Skipping...")
        else:
            try:
                # Adding seasonality with SARIMAX
                seasonal_order = (1, 0, 1, 52)  
                model = SARIMAX(y_train, order=(5, 1, 0), seasonal_order=seasonal_order)
                fitted_model = model.fit()

                # Forecast for y_test weeks
                forecast_index = y_test.index  # Align forecast with test set index
                y_pred = fitted_model.forecast(steps=len(forecast_index))
                # y_pred = best_model.predict(X_test)

                # Convert predictions to a pandas Series
                y_pred_series = pd.Series(y_pred, index=y_test.index)

                # Calculate error metrics
                mse = mean_squared_error(y_test, y_pred_series)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred_series)
                mape = np.mean(np.abs((y_test - y_pred_series) / y_test)) * 100

                # Calculate MASE
                naive_forecast = y_train.diff().dropna()
                mase = mae / np.mean(np.abs(naive_forecast))

                group_accuracy.append({
                    'SKU_Number': '425',
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape,
                    'mase': mase
                })

                group_prediction['Kids'] = y_pred_series

            except Exception as e:
                print(f"Could not fit model for {'326'} {'Shoes'} due to: {e}")

        return pd.DataFrame(group_accuracy), group_prediction

    def create_prediction_dataframe(self, prediction):
        prediction_list = []

        for group, pred_series in prediction.items():
            for forecast_week, forecast in pred_series.items():
                prediction_list.append({
                    'Store': 'Dallas',
                    'SKU_Number': group,
                    'Forecast_Week': forecast_week,
                    'Forecasted_Units_Sold': forecast,
                })

        return pd.DataFrame(prediction_list)

    def run_forecasting(self):
        self.load_data()
        combined_metrics = []
        combined_prediction = []

        # for store in self.data['Store_Number'].unique():
        #     for group_column in ['SKU_Number', 'Department', 'Category']:
        group_metrics, group_prediction = self.fit_forecast()
        prediction_df = self.create_prediction_dataframe(group_prediction)

        combined_metrics.append(group_metrics)
        combined_prediction.append(prediction_df)

        all_metrics_df = pd.concat(combined_metrics, ignore_index=True)
        all_predictions_df = pd.concat(combined_prediction, ignore_index=True)

        return all_predictions_df, all_metrics_df


def main():
    file_path = '/Users/anabellaisaro/Documents/Documents - Anabella’s MacBook Pro/Northwestern/Projects/Deloitte/forecast/data/'
    # input_file_path = os.path.join(os.path.dirname(file_path), 'Capstone_Forecasting_Data_updated.csv')
    input_file_path = os.path.join(os.path.dirname(file_path), 'category_sku_level.csv')
    data = pd.read_csv(input_file_path)

    es_model = ExponentialSmoothingModel(data)
    all_prediction_df, all_metric_df = es_model.run_forecasting()

    output_file_path = os.path.join(os.path.dirname(file_path), 'forecasted_sales_sal_cat.csv')

    metric_output_file_path = os.path.join(os.path.dirname(file_path), 'metrics-sal_cat.csv')

    all_prediction_df.to_csv(output_file_path, index=False)
    all_metric_df.to_csv(metric_output_file_path, index=False)


if __name__ == "__main__":
    main()
