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
        years = [2020, 2021, 2023, 2024, 2025]
        self.data['Store_Location'] = self.data['Store_SKU'].str.split('_').str[1]
        self.data['Store_Location'] = self.data['Store_Location'].astype(int, errors='ignore')  # Convert to int
        self.data['Store_Number'] = self.data['Store_SKU'].str.extract(r'(\d+)')
        self.data['Store_Number'] = self.data['Store_Number'].astype(int, errors='ignore')  # Convert to int
        self.data['SKU_Number'] = self.data['SKU'].str.extract(r'(\d+)')
        self.data['SKU_Number'] = self.data['SKU_Number'].astype(int, errors='ignore')  # Convert to int
        self.data['Week'] = pd.to_datetime(self.data['Week']).dt.tz_localize(None)
        us_holidays = holidays.CountryHoliday('US', years=years)
        holiday_dates = pd.to_datetime(list(us_holidays.keys()))
        self.data['is_week_before_holiday'] = self.data['Week'].apply(
            lambda x: int(any((0 < (holiday - x).days <= 7) for holiday in holiday_dates))
        )
    def fit_forecast(self):
        group_accuracy = []
        group_prediction = {}
        store_data = self.data

        # Filter for specific store, SKU, and Category
        group_data = store_data[
            (store_data['Store_Location'] == 'Chicago') &
            # (store_data['Category'] == 'Womens') &
            # (store_data['Department'] == 'Grocery') &
            (store_data['SKU_Number'] == 227) 
        ]

        group_data.set_index('Week', inplace=True)
        group_data = group_data.resample('W').sum()
        start_week='2023-07-01'
        y = group_data['UNITS_SOLD']
        start_index = group_data.index[group_data.index >= pd.Timestamp(start_week)]
        if start_index.empty:
            print(f"No data available starting from {start_week}.")
            return pd.DataFrame(group_accuracy), group_prediction

        # train_end_index = group_data.index.get_loc(start_index[0])  # Get the index for splitting
        # y_train = y[:train_end_index]  # Training data up to (but not including) start_week
        # y_test = y[train_end_index:]  # Test data starts from start_week
         # Create additional features
        group_data['Week_Number'] = np.arange(len(group_data))  
        X = group_data[['Week_Number', 'is_week_before_holiday']]  
        y = group_data['UNITS_SOLD']

        train_size = int(len(group_data) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        if len(y_train) < 10:
            print("y_train has fewer than 10 elements. Skipping...")
        else:
            try:
            # Create polynomial features
                # Define parameter grid
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 5],
                    'max_features': ['sqrt', 'log2', 0.8]
                }

                # Initialize RandomForestRegressor
                model = RandomForestRegressor(random_state=42)

                # Set up GridSearchCV
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )

                # Fit grid search to training data
                grid_search.fit(X_train, y_train)

                # Use the best estimator for predictions
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)

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
    input_file_path = os.path.join(os.path.dirname(file_path), 'Capstone_Forecasting_Data_updated.csv')
    data = pd.read_csv(input_file_path)

    es_model = ExponentialSmoothingModel(data)
    all_prediction_df, all_metric_df = es_model.run_forecasting()

    output_file_path = os.path.join(os.path.dirname(file_path), 'forecasted_sales_exp_smoothing_sku.csv')
    metric_output_file_path = os.path.join(os.path.dirname(file_path), 'metrics-exp_smoothing_sku.csv')

    all_prediction_df.to_csv(output_file_path, index=False)
    all_metric_df.to_csv(metric_output_file_path, index=False)


if __name__ == "__main__":
    main()
