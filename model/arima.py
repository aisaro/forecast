import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
import matplotlib.pyplot as plt

# To ignore warnings from ARIMA
warnings.filterwarnings("ignore")

class ARIMAModel:
    def __init__(self, p, d, q):

        self.p = p
        self.d = d
        self.q = q
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
    def plot_predictions(self, df, target_col, date_col):
        forecast = model_fit.forecast(steps=208)  
        print('Forecasted Units Sold for next 4 weeks:', forecast)

        plt.figure(figsize=(10, 6))
        plt.plot(weekly_units_sold, label='Actual Units Sold', color='blue')
        plt.plot(forecast.index, forecast, label='Forecasted Units Sold', color='red')
        plt.xlabel('Week')
        plt.ylabel('Units Sold')
        plt.title('Units Sold Forecast using ARIMA')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    file_path = 'FILE_PATH'
    arima_model = ARIMAModel(p=1, d=1, q=1)
    data = arima_model.load_data(file_path)
    data.set_index('Week', inplace=True)

    weekly_units_sold = data.resample('W').sum()['Units_Sold']
    adf_result = adfuller(weekly_units_sold)
    print('ADF Statistic:', adf_result[0])
    print('p-value:', adf_result[1])
    if adf_result[1] > 0.05:
        weekly_units_sold_diff = weekly_units_sold.diff().dropna()
    else:
        weekly_units_sold_diff = weekly_units_sold
    model = ARIMA(weekly_units_sold_diff.dropna(), order=(1, 1, 1))  #
    model_fit = model.fit()

    print(model_fit.summary())

    
