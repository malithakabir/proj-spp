
import numpy as np
from sklearn.preprocessing import StandardScaler

def print_data_split(df_train, df_test, df_predict):
    print("Training data starts at", df_train.index[0])
    print("Training - test split at", df_train.index[-1])
    print("Testing data ends at", df_test.index[-1])
    print()
    print("prediction data starts at", df_predict.index[0])
    print("prediction data ends at", df_predict.index[-1])

class Preprocess:
    def __init__(self):
        self.isnadrop = False
    def set_df(self, df):
        self.df = df
    def set_test_size(self, val):
        self.test_size = val
    def set_rolling_window(self, val):
        """
        To fit the model
        """
        self.rolling_window = val
    def set_forecast_horizon(self, val):
        """
        To forecast data
        """
        self.forecast_horizon = val
    def set_sample_size_mt(self, val):
        self.sample_size_mt = val
    def dropna(self):
        self.isnadrop = True
        self.df = self.df.dropna(axis = 0).copy()
    def get_idx_tt_split(self, test_size, sample_size_mt):
        # split the data into train and test data.
        idx_tt_split = round((1-test_size) * sample_size_mt)
        return idx_tt_split
    def generate_train_test_predict_split(self):
        """
        Required:
            df
            rolling_window
            test_size
        Returns: dict [df_train, df_test, df_predict]
        """
        if self.isnadrop:
            df_predict = self.df.iloc[-self.rolling_window:] # for future forecasting use
            df_train_test = self.df.iloc[:-self.rolling_window] # for model training and testing use
            
            self.set_sample_size_mt(df_train_test.shape[0])
            self.idx_tt_split = self.get_idx_tt_split(self.test_size, self.sample_size_mt)
            
            df_train = df_train_test.iloc[:self.idx_tt_split, :]
            df_test = df_train_test.iloc[self.idx_tt_split:, :]
            
            # print results
            print_data_split(df_train, df_test, df_predict)

            return {'df_train': df_train, 'df_test': df_test, 'df_predict': df_predict}
        else:
            print('Drop NA first using dropna() method')
    def calculate_sequence_length(self):
        self.sequence_length = self.rolling_window + self.forecast_horizon
        
    def normalise_dataframe(self, df, step=1, standard_norm=True):
        normalised_data = []
        scalers = []
        #form a sliding window of size 'sequence_length', until the data is exhausted
        for index in range(0, df.shape[0] - self.sequence_length + 1, step):
            window = df[index: index + self.sequence_length]
            scaler = StandardScaler()
            scaler.fit(window)
            normalised_data.append(scaler.transform(window))
            scalers.append(scaler)
        return {'normalised_data': normalised_data, 'scalers': scalers}
    def prepare_feature_and_label(self, data_list):
        arr = np.array(data_list)
        
        # shape: sample, timeframe, features
        features = arr[:, :-self.forecast_horizon, 1:] # features: everything except the first col
        
        labels = arr[:, -self.forecast_horizon: , 0] # labels: the first col only
        # reshape label to sample, timeframe, label
        labels = np.reshape(labels, (labels.shape[0], labels.shape[1], 1))

        return {'data': arr,'features': features, 'labels': labels}
    
