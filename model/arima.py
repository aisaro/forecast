import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings

# To ignore warnings from ARIMA
warnings.filterwarnings("ignore")

class ARIMAModel:
    def __init__(self, p, d, q):

        self.p = p
        self.d = d
        self.q = q
        self.model = None
        self.fitted_model = None
    
    def load_data(self, file_path, column_name):
        data = pd.read_csv(file_path)
        if column_name not in data.columns:
            raise ValueError(f"Column '{column_name}' not found in the file.")
        return data[column_name]
    
    def fit(self, data):

        self.model = ARIMA(data, order=(self.p, self.d, self.q))
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
    arima_model = ARIMAModel(p=1, d=1, q=1)
    data = arima_model.load_data(file_path, column_name)

    arima_model.fit(data)
    print(arima_model.summary())
    forecast = arima_model.predict(steps=5)
