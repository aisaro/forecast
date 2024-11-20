import os
import holidays
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class RandomForestTimeSeries:
    def __init__(self, data, lags=5, n_estimators=100):
        self.data = data
        self.lags = lags
        self.n_estimators = n_estimators

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
        # # Ensure 'Department' and 'Category' have no missing or invalid data
        # self.data['Department'] = self.data['Department'].fillna("Unknown").astype(str)
        # self.data['Category'] = self.data['Category'].fillna("Unknown").astype(str)

        # # Label Encoding for Department and Category
        # label_encoder_dept = LabelEncoder()
        # label_encoder_cat = LabelEncoder()
        # self.data['Department'] = label_encoder_dept.fit_transform(self.data['Department'])
        # self.data['Category'] = label_encoder_cat.fit_transform(self.data['Category'])
        # # Select numeric columns
        # numeric_columns = self.data.select_dtypes(include=[np.number]).columns

        # # Replace infinite values with NaN
        # self.data[numeric_columns].replace([np.inf, -np.inf], np.nan, inplace=True)

        # # Fill missing values (NaN) with a default value (e.g., 0 or column mean)
        # for col in numeric_columns:
        #     self.data[col].fillna(self.data[col].mean(), inplace=True)

        # # Optionally: Limit excessively large values (outliers)
        # for col in numeric_columns:
        #     self.data[col] = np.clip(self.data[col], a_min=None, a_max=1e5)  # Adjust 1e5 as needed

        # print("Checking for invalid values...")
        # print("NaN values:", self.data[numeric_columns].isna().sum().sum())
        # print("Infinite values:", np.isinf(self.data[numeric_columns]).sum().sum())
        # print("Max value:", self.data[numeric_columns].max(numeric_only=True))
    def create_lag_features(self, df, target_col):
        """
        Create lag-based features for a time series.
        """
        for lag in range(1, self.lags + 1):
            df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
        df.dropna(inplace=True)
        return df
    
    def fit_forecast(self, group_column, store):
        group_metrics = []
        group_prediction = {}
        store_data = self.data[self.data['Store_Number'] == store]
        unique_groups = ['Home']  # Example group
        unique_skus = self.data['SKU_Number'].unique()

        for sku in unique_skus:
            sku_data = store_data[store_data['SKU_Number'] == sku].copy()  # Filter for specific SKU
            for group in unique_groups:
                group_data = sku_data[sku_data[group_column] == group].copy()
                group_data.set_index('Week', inplace=True)
                group_data = group_data.resample('W').sum()
            # group_data.set_index('Elapsed_Days', inplace=True)
            # group_data = group_data.resample('W').sum()

            # Create lag features
            group_data = self.create_lag_features(group_data, 'UNITS_SOLD')

            # Split data into training and testing sets
            train_size = int(len(group_data) * 0.8)
            X = group_data.drop(columns=['UNITS_SOLD'])
            y = group_data['UNITS_SOLD']
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            if len(y_train) < self.lags:  # Skip if not enough data points
                continue

            # Train Random Forest Regressor
            model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            group_metrics.append({
                group_column: group,
                'Store': store,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae
            })

            # Save predictions
            group_prediction[group] = y_pred

        return pd.DataFrame(group_metrics), group_prediction

    def create_prediction_dataframe(self, store, prediction):
        prediction_list = []

        for group, pred in prediction.items():
            for week_offset in range(len(pred)):
                prediction_list.append({
                    'Store': store,
                    'SKU_Number': group,
                    # group_column: group,
                    'Forecast_Week': pd.Timestamp.now() + pd.DateOffset(weeks=week_offset + 1),
                    'Forecasted_Units_Sold': pred[week_offset],
                })

        return pd.DataFrame(prediction_list)

    def run_forecasting(self):
        self.load_data()
        combined_metrics = []
        combined_predictions = []

        unique_stores = self.data['Store_Number'].unique()
        # for store in unique_stores:
        for group_column in ['Department', 'Category']:
            group_metrics, group_prediction = self.fit_forecast(group_column, 5)
            prediction_df = self.create_prediction_dataframe(5, group_column, group_prediction)

            combined_metrics.append(group_metrics)
            combined_predictions.append(prediction_df)

        all_metrics_df = pd.concat(combined_metrics, ignore_index=True)
        all_predictions_df = pd.concat(combined_predictions, ignore_index=True)

        return all_predictions_df, all_metrics_df


# Main Execution
def main():
    file_path = '/Users/anabellaisaro/Documents/Documents - Anabella’s MacBook Pro/Northwestern/Projects/Deloitte/forecast/data/'
    input_file_path = os.path.join(os.path.dirname(file_path), 'Capstone_Forecasting_Data_updated.csv')
    data = pd.read_csv(input_file_path)

    rf_model = RandomForestTimeSeries(data, lags=5, n_estimators=100)
    all_prediction_df, all_metrics_df = rf_model.run_forecasting()

    # Save results to CSV
    output_file_path = os.path.join(os.path.dirname(file_path), 'forecasted_sales_RF.csv')
    metric_output_file_path = os.path.join(os.path.dirname(file_path), 'metrics_RF.csv')

    all_prediction_df.to_csv(output_file_path, index=False)
    all_metrics_df.to_csv(metric_output_file_path, index=False)


if __name__ == "__main__":
    main()
