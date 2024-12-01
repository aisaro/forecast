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
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
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
            (store_data['Store_Location'] == 'Baltimore') &
            # (store_data['Category'] == 'Womens') &
            # (store_data['Department'] == 'Grocery') &
            (store_data['SKU_Number'] == 335) 
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
        X = group_data[['Week_Number']]  
        y = group_data['UNITS_SOLD']

        train_size = int(len(group_data) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        if len(y_train) < 10:
            print("y_train has fewer than 10 elements. Skipping...")
        else:
            try:
                # Define base models
                random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
                gradient_boosting = GradientBoostingRegressor(n_estimators=100, random_state=42)
                ridge = Ridge(alpha=1.0)
                svr = SVR(kernel='rbf', C=1.0)

                # ## 1. Stacking Regressor ###
                # stacking_regressor = StackingRegressor(
                #     estimators=[
                #         ('rf', random_forest),
                #         ('gb', gradient_boosting),
                #         ('ridge', ridge)
                #     ],
                #     final_estimator=SVR(kernel='linear')
                # )

                # # Fit and predict with Stacking Regressor
                # stacking_regressor.fit(X_train, y_train)
                # y_pred = stacking_regressor.predict(X_test)

                # # Convert predictions to a pandas Series
                # y_pred_series = pd.Series(y_pred, index=y_test.index)
                # # Fit and predict with Stacking Regressor
                # stacking_regressor.fit(X_train, y_train)
                # stacking_pred = stacking_regressor.predict(X_test)
                # stacking_mse = mean_squared_error(y_test, stacking_pred)
                # print(f"Stacking Regressor MSE: {stacking_mse:.4f}")

                ### 2. Voting Regressor ###
                voting_regressor = VotingRegressor(
                    estimators=[
                        ('rf', random_forest),
                        ('gb', gradient_boosting),
                        ('ridge', ridge)
                    ]
                )

                # Fit and predict with Voting Regressor
                voting_regressor.fit(X_train, y_train)
                y_pred = voting_regressor.predict(X_test)
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
    input_file_path = os.path.join(os.path.dirname(file_path), 'store_sku_level.csv')
    data = pd.read_csv(input_file_path)

    es_model = ExponentialSmoothingModel(data)
    all_prediction_df, all_metric_df = es_model.run_forecasting()

    output_file_path = os.path.join(os.path.dirname(file_path), 'forecasted_sales_exp_smoothing_sku.csv')
    metric_output_file_path = os.path.join(os.path.dirname(file_path), 'metrics-exp_smoothing_sku.csv')

    all_prediction_df.to_csv(output_file_path, index=False)
    all_metric_df.to_csv(metric_output_file_path, index=False)


if __name__ == "__main__":
    main()
