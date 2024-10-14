import numpy as np
import pandas as pd

class MovingAverageModel:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.data = None
        
    def load_data(self, file_path, column_name):
        data = pd.read_csv(file_path)
        if column_name not in data.columns:
            raise ValueError(f"Column '{column_name}' not found in the file.")
        self.data = data[column_name]
        return self.data
    
    def moving_average(self, data):
        return data.rolling(window=self.window_size).mean()
    
    def predict(self, steps=1):
        if self.data is None:
            raise ValueError("No data loaded. Load data using load_data() method first.")
        
        predictions = []
        last_obs = self.data[-self.window_size:]  # Most recent `window_size` observations
        
        for _ in range(steps):
            ma_value = last_obs.mean()
            predictions.append(ma_value)
            last_obs = np.roll(last_obs, -1)
            last_obs[-1] = ma_value
        
        return predictions

if __name__ == "__main__":
    file_path = '/Users/anabellaisaro/Documents/Documents - Anabella’s MacBook Pro/Northwestern/Projects/Deloitte/forecast/data/Forecasting_Schema_Example_20241007.csv'
    column_name = ""
    ma_model = MovingAverageModel(window_size=3)
    data = ma_model.load_data(file_path, column_name)
    moving_avg = ma_model.moving_average(data)
    forecast = ma_model.predict(steps=5)
