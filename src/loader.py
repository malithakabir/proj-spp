import os
import numpy as np
import pandas as pd
import re

# import talib as tb
from sklearn.preprocessing import StandardScaler
import pandas_datareader as pdr


class DataLoader:
    def __init__(self, ticker, dirname=None, preprocessed=None):
        print("Instantiating data loader")
        if ticker is None:
            print("Ticker is missing")
        else:
            self.ticker = ticker.upper()
        if dirname is not None:
            self.datadir = dirname

        self.preprocessed = None
        if preprocessed is not None:
            self.preprocessed = preprocessed
        self.filename_processed = None
        self.df_processed = None
        self.filename_raw = None
        self.df_raw = None

    def set_datadir(self, dirname):
        self.datadir = dirname

    def read_local(self, preprocessed=True):
        if preprocessed:
            fname = "{}Data_for_{}.csv".format(self.datadir, self.ticker)
            print("Reading from: {}".format(fname))
            self.filename_processed = fname
            self.df_processed = pd.read_csv(
                self.filename_processed, parse_dates=True, index_col=0
            )
        else:
            fname = "{}{}.csv".format(self.datadir, self.ticker)
            print("Reading from: {}".format(fname))
            self.filename_raw = fname
            self.df_raw = pd.read_csv(
                self.filename_raw, header=0, index_col="Date", parse_dates=True
            )

    #     # if tf == None:
    #     #     self.df = data
    #     # else:
    #     #     self.df = self.gen_TimeFrame(data, tf)
    #     # print("The data is from {} to {}".format(self.df.index[0], self.df.index[-1]))
    # def gen_TimeFrame(self, data, tf):
    #     tf_num = int(re.search(r'\d+', tf).group())   # how many
    #     tf_type = str.capitalize(tf[-1])  #  select 'Y', 'M', 'D'
    #     df2 = data[data.index > (data.index[-1] - np.timedelta64(1,tf_type)*tf_num)]
    #     return df2
    # def export_data(self):
    #     self.df_out = self.prepare_technical()
    #     self.df_out.to_csv('Data_for_{}.csv'.format(self.stock), sep=',')


from keras.models import load_model


class ModelLoder:
    def __init__(self) -> None:
        self.ticker = None
        self.modeldir = None
        self.forecast_horizon = None
        self.rolling_window = None
        self.model = None

    def set_model_dir(self, dirname):
        self.modeldir = dirname

    def set_ticker(self, ticker):
        self.ticker = ticker

    def set_forecast_horizon(self, val):
        self.forecast_horizon = val

    def set_rolling_window(self, val):
        self.rolling_window = val

    def read_model_local(self):
        model_name = "{}LSTM_{}_{}_{}.h5".format(
            self.modeldir, self.ticker, self.rolling_window, self.forecast_horizon
        )
        print("Loading model from {}".format(model_name))
        self.model = load_model(model_name)

    def get_model_list(self):
        """
        Incomplete implemenation
        """
        return
