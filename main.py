"""
TODO: This file will have command line implementations.
"""

from src.loader import DataLoader, ModelLoder
from src.preprocessing import Preprocess
from src.model import CustomModel
from src.postprocessing import invert_scale_N_feature

ticker = 'IPGP'
rolling_window = 20
forecast_horizon = 20
test_size = 0.3

data = DataLoader(ticker='IPGP')
data.set_datadir(dirname='data/')
data.read_local(preprocessed=True)

print('printing first 5 rows')
print(data.df_processed.head(5))

print()
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

# print()
# print('last 5 rows from forecast split')
# print(data_split['df_predict'].tail(5))

# set forecast_horizon
preprocessor.set_forecast_horizon(forecast_horizon)
preprocessor.calculate_sequence_length()

ndata_train = preprocessor.normalise_dataframe(df=data_split['df_train'], step=1, standard_norm=True)
scalers_train = ndata_train['scalers']
pdata_train = preprocessor.prepare_feature_and_label(data_list=ndata_train['normalised_data'])
train_data = pdata_train['data']
X_train = pdata_train['features']
y_train = pdata_train['labels']

ndata_test = preprocessor.normalise_dataframe(df=data_split['df_test'], step=1, standard_norm=True)
scalers_test = ndata_test['scalers']
pdata_test = preprocessor.prepare_feature_and_label(data_list=ndata_test['normalised_data'])
test_data = pdata_test['data']
X_test = pdata_test['features']
y_test = pdata_test['labels']

print()
print('train_data.shape', train_data.shape)
print('X_train.shape', X_train.shape)
print('y_train.shape', y_train.shape)
print('y_train.squeeze().shape', y_train.squeeze().shape)

print()
print('test_data.shape', test_data.shape)
print('X_test.shape', X_test.shape)
print('y_test.shape', y_test.shape)
print('y_test.squeeze().shape', y_test.squeeze().shape)

# Model Loader
model_loader = ModelLoder()
model_loader.set_model_dir(dirname='model/')
model_loader.set_ticker(ticker=ticker)
model_loader.set_rolling_window(rolling_window)
model_loader.set_forecast_horizon(forecast_horizon)

model_loader.read_model_local()
model = model_loader.model

print(model.summary())

# run predictions
predicts = model.predict(X_test) # shape: (sample, output)
y_test = y_test.squeeze()

# print()
# calculate error

model = CustomModel()

drop_rate=0.1
latent_n=400
feature_n= X_train.shape[2]

model = model.build_model(
    input_n=rolling_window,
    output_n=forecast_horizon,
    drop_rate=drop_rate,
    latent_n=latent_n,
    feature_n=feature_n
    )

