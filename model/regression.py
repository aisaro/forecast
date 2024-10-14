import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class LinearRegressionModel:
    def __init__(self, lags=1):

        self.lags = lags
        self.model = LinearRegression()
        
    def load_data(self, file_path, column_name):
        data = pd.read_csv(file_path)
        if column_name not in data.columns:
            raise ValueError(f"Column '{column_name}' not found in the file.")
        return data[column_name]
    
    def create_lagged_features(self, data):
        df = pd.DataFrame(data)
        for i in range(1, self.lags + 1):
            df[f'lag_{i}'] = df[0].shift(i)
        df = df.dropna()
        X = df.drop(0, axis=1).values
        y = df[0].values
        return X, y
    
    def fit(self, data):
        X, y = self.create_lagged_features(data)
        self.model.fit(X, y)
        
    def predict(self, data, steps=1):
        predictions = []
        last_obs = data[-self.lags:]  
        
        for _ in range(steps):
            X_pred = last_obs.reshape(1, -1)
            pred = self.model.predict(X_pred)
            predictions.append(pred[0])
            last_obs = np.roll(last_obs, -1)  
            last_obs[-1] = pred  
            
        return predictions
    
    def summary(self):
        return {"Coefficients": self.model.coef_, "Intercept": self.model.intercept_}

if __name__ == "__main__":
    file_path = '/Users/anabellaisaro/Documents/Documents - Anabella’s MacBook Pro/Northwestern/Projects/Deloitte/forecast/data/Forecasting_Schema_Example_20241007.csv'
    column_name = ""
    lr_model = LinearRegressionModel(lags=3)
    data = lr_model.load_data(file_path, column_name)
    lr_model.fit(data)
    forecast = lr_model.predict(data, steps=5)
