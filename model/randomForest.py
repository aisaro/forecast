import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import holidays
from datetime import datetime
import os
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

class RegressionForecastingModel:
    def __init__(self, data):
        self.index_data = None
        self.data = data

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

    def fit_forecast(self, group_column, store):
        group_accuracy = []
        group_prediction = {}
        store_data = self.data[self.data['Store_Number'] == store]
        unique_groups = ['Shoes']  # Example group
        unique_skus = self.data['SKU_Number'].unique()

        for sku in unique_skus:
            sku_data = store_data[store_data['SKU_Number'] == sku].copy()  # Filter for specific SKU
            for group in unique_groups:
                group_data = sku_data[sku_data[group_column] == group].copy()
                group_data.set_index('Week', inplace=True)
                group_data = group_data.resample('W').sum()

                # Create additional features
                group_data['Week_Number'] = np.arange(len(group_data))  # Sequential number for weeks
                X = group_data[['Week_Number', 'is_week_before_holiday']]  # Features: Week number and holiday flag
                y = group_data['UNITS_SOLD']

                train_size = int(len(group_data) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]

                if len(y_train) < 10:
                    continue

                try:
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [10, 20, 30, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 5],
                        'max_features': ['sqrt', 'log2', 0.8]
                    }

                    model = RandomForestRegressor(random_state=42)
                    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
                    grid_search.fit(X_train, y_train)
                    # model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                    y_pred_series = pd.Series(y_pred, index=y_test.index)

                    mse = mean_squared_error(y_test, y_pred_series)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred_series)
                    mape = np.mean(np.abs((y_test - y_pred_series) / y_test)) * 100

                    group_accuracy.append({'Store': store, 'SKU_Number': sku, 'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape})
                    group_prediction[sku] = y_pred_series
                except Exception as e:
                    print(f"Could not fit model for {group_column} {sku} in Store {store} due to: {e}")

        return pd.DataFrame(group_accuracy), group_prediction

    def create_prediction_dataframe(self, store, group_column, prediction):
        prediction_list = []

        for group, pred_series in prediction.items():
            for forecast_week, forecast in pred_series.items():
                prediction_list.append({
                    'Store': store,
                    'SKU_Number': group,
                    'Forecast_Week': forecast_week,
                    'Forecasted_Units_Sold': forecast,
                })

        return pd.DataFrame(prediction_list)

    def run_forecasting(self):
        self.load_data()
        combined_metrics = []
        combined_prediction = []
        #TODO Uncomment aline 102 and replace 5 with store
        # for store in self.data['Store_Number'].unique():
        for group_column in ['Department', 'Category']:
            group_metrics, group_prediction = self.fit_forecast(group_column, 5)
            prediction_df = self.create_prediction_dataframe(5, group_column, group_prediction)

            combined_metrics.append(group_metrics)
            combined_prediction.append(prediction_df)

        all_metrics_df = pd.concat(combined_metrics, ignore_index=True)
        all_predictions_df = pd.concat(combined_prediction, ignore_index=True)

        return all_predictions_df, all_metrics_df


def main():
    file_path = '/Users/anabellaisaro/Documents/Documents - Anabella’s MacBook Pro/Northwestern/Projects/Deloitte/forecast/data/'
    input_file_path = os.path.join(os.path.dirname(file_path), 'Capstone_Forecasting_Data_updated.csv')
    data = pd.read_csv(input_file_path)

    regression_model = RegressionForecastingModel(data)
    all_prediction_df, all_metric_df = regression_model.run_forecasting()

    output_file_path = os.path.join(os.path.dirname(file_path), 'category/forecasted_sales_RF_shoes_grid.csv')
    metric_output_file_path = os.path.join(os.path.dirname(file_path), 'category/metrics_RF_shoes_grid.csv')

    all_prediction_df.to_csv(output_file_path, index=False)
    all_metric_df.to_csv(metric_output_file_path, index=False)


if __name__ == "__main__":
    main()
