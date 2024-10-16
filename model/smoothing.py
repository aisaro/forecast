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
        
    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        data['Store_Number'] = data['Store'].str.split('_').str[1]
        data['Department_Number'] = data['DEPARTMENT'].str.extract(r'(\d+)')
        data['SKU_Number'] = data['SKU'].str.extract(r'(\d+)')
        data['Category_Number'] = data['CATEGORY'].str.extract(r'(\d+)')
        data['Week'] = pd.to_datetime(data['Week'])
        return data
    
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
    file_path = 'FILE_PATH'
    exp_model = ExponentialSmoothingModel(trend='add', seasonal='add', seasonal_periods=12)
    data = exp_model.load_data(file_path)
    exp_model.fit(data)
    forecast = exp_model.predict(steps=5)
