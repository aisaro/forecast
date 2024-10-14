import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

# To ignore warnings
warnings.filterwarnings("ignore")

class ExponentialSmoothingModel:
    def __init__(self, trend=None, seasonal=None, seasonal_periods=None):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None
        self.fitted_model = None
        
    def load_data(self, file_path, column_name):
        data = pd.read_csv(file_path)
        if column_name not in data.columns:
            raise ValueError(f"Column '{column_name}' not found in the file.")
        return data[column_name]
    
    def fit(self, data):
        self.model = ExponentialSmoothing(data, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods)
        self.fitted_model = self.model.fit()
        print("Model fitted successfully.")
        
    def predict(self, steps=1):
        if self.fitted_model is None:
            raise ValueError("The model has not been fitted yet.")
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast
    
    def summary(self):

        if self.fitted_model is None:
            raise ValueError("The model has not been fitted yet.")
        return self.fitted_model.summary()

if __name__ == "__main__":
    file_path = '/Users/anabellaisaro/Documents/Documents - Anabella’s MacBook Pro/Northwestern/Projects/Deloitte/forecast/data/Forecasting_Schema_Example_20241007.csv'
    column_name = ""
    exp_model = ExponentialSmoothingModel(trend='add', seasonal='add', seasonal_periods=12)
    data = exp_model.load_data(file_path, column_name)
    exp_model.fit(data)
    forecast = exp_model.predict(steps=5)
