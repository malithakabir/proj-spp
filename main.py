"""
TODO: This file will have command line implementations.
"""

from src.loader import DataLoader, ModelLoder
from src.preprocessing import Preprocess

ticker = 'IPGP'
rolling_window = 20
forecast_horizon = 20
test_size = 0.3

print('Instantiating data loader')
data = DataLoader(ticker='IPGP')
data.set_datadir(dirname='data/')
data.read_local(preprocessed=True)

print('printing first 5 rows')
print(data.df_processed.head(5))

print()
print('Entering to preprocessor')
preprocessor = Preprocess()
preprocessor.set_df(data.df_processed)
preprocessor.dropna()

# keep data points for forecasting
preprocessor.set_rolling_window(rolling_window)
# percentage of test data 
preprocessor.set_test_size(test_size)

data_split = preprocessor.generate_train_test_predict_split()
print()
print('first 5 rows from training split')
print(data_split['df_train'].head(5))

print()
print('first 5 rows from test split')
print(data_split['df_test'].head(5))

print()
print('first 5 rows from forecast split')
print(data_split['df_predict'].head(5))

print()
print('last 5 rows from forecast split')
print(data_split['df_predict'].tail(5))
# preprocessor.set_forecast_horizon(20)


# Model Loader
model_loader = ModelLoder()
model_loader.set_model_dir(dirname='model/')
model_loader.set_ticker(ticker=ticker)
model_loader.set_rolling_window(rolling_window)
model_loader.set_forecast_horizon(forecast_horizon)

model_loader.read_model_local()
model = model_loader.model

print(model.summary())
