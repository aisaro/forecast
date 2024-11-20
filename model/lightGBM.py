import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

class LightGBMTimeSeries:
    def __init__(self, lags, params=None):
        self.lags = lags
        self.params = params if params is not None else {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        self.model = None
        self.features = None

    def create_lag_features(self, df, target_col):
        for lag in self.lags:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        return df

    def add_time_features(self, df, date_col):
        df['day_of_week'] = df['Week'].dt.dayofweek
        df['month'] = df['Week'].dt.month
        return df

    def train(self, df, target_col, date_col, test_size=0.2):
        # Create lag and time-based features
        df = self.create_lag_features(df, target_col)
        df = self.add_time_features(df, date_col)
        df = df.dropna()  # Drop rows with NaN values due to lag

        # Define features and target
        self.features = [f'lag_{lag}' for lag in self.lags] + ['day_of_week', 'month']
        X = df[self.features]
        y = df[target_col]

        # Split data into train and test sets
        train_size = int((1 - test_size) * len(df))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test)

        # Train the model
        self.model = lgb.train(self.params, train_data, valid_sets=[test_data], early_stopping_rounds=50, num_boost_round=1000)

        # Predictions and performance on test set
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Test RMSE: {rmse}")

    def predict(self, df, date_col):

        df = self.create_lag_features(df, 'Sales')
        df = self.add_time_features(df, date_col)
        X = df[self.features].dropna()  # Drop any rows with NaN values due to lags

        predictions = self.model.predict(X)
        return predictions

    def plot_predictions(self, df, target_col, date_col):
        # Make predictions
        predictions = self.predict(df, date_col)

        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.plot(df[date_col][-len(predictions):], df[target_col][-len(predictions):], label='Actual', marker='o')
        plt.plot(df[date_col][-len(predictions):], predictions, label='Predicted', marker='x')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title('Actual vs Predicted Sales')
        plt.legend()
        plt.show()

