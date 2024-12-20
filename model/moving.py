import numpy as np
import pandas as pd

class MovingAverageModel:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.data = None
        
    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        data['Store_Number'] = data['Store'].str.split('_').str[1]
        data['Department_Number'] = data['DEPARTMENT'].str.extract(r'(\d+)')
        data['SKU_Number'] = data['SKU'].str.extract(r'(\d+)')
        data['Category_Number'] = data['CATEGORY'].str.extract(r'(\d+)')
        data['Week'] = pd.to_datetime(data['Week'])
        return data
    
    def moving_average(self, data):
        return data.rolling(window=self.window_size).mean()
    
    def predict(self, steps=1):
        if self.data is None:
            raise ValueError("No data loaded. Load data using load_data() method first.")
        
        predictions = []
        last_obs = self.data[-self.window_size:] 
        
        for _ in range(steps):
            ma_value = last_obs.mean()
            predictions.append(ma_value)
            last_obs = np.roll(last_obs, -1)
            last_obs[-1] = ma_value
        
        return predictions

if __name__ == "__main__":
    file_path = 'FILE_PATH'
    ma_model = MovingAverageModel(window_size=3)
    data = ma_model.load_data(file_path)
    moving_avg = ma_model.moving_average(data)
    forecast = ma_model.predict(steps=5)
