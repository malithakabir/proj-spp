import os
import numpy as np
import pandas as pd
import re
import json
# import talib as tb
from sklearn.preprocessing import StandardScaler
from pandas_datareader import data as pdr
from datetime import datetime
import yfinance as yf
yf.pdr_override()


class DataLoader:
    def __init__(self, ticker, dirname=None):
        print("Instantiating data loader")
        if ticker is None:
            print("Ticker is missing")
        else:
            self.ticker = ticker.upper()
        if dirname is not None:
            self.datadir = dirname

        self.isTechnical = None
        self.filename_processed = None
        self.df_processed = None
        self.filename_raw = None
        self.df_raw = None
        
        with open('registration.json', encoding='UTF-8') as f:
            self.registraion = json.load(f)

        self.dateformat = '%Y-%m-%d'
        self.remote_data = None

    def set_datadir(self, dirname):
        self.datadir = dirname

    def read_local(self, isTechnical=True, latest=True):
        self.isTechnical = isTechnical
        rdata = self.registration.get(self.ticker, None).get('data', None)
        if latest:
            key = str(max([int(i) for i in rdata.keys()]))
        else:
            key = "0" 
        
        if isTechnical:
            fname = rdata.get(key, None).get('technical_analysis_filename', None)
            # fname = "{}{}_TA_V1.csv".format(self.datadir, self.ticker)
            fpath = self.datadir + fname
            print("Reading from: {}".format(fpath))
            self.filename_processed = fpath
            self.df_processed = pd.read_csv(
                self.filename_processed, parse_dates=True, index_col=0
            )
        else:
            fname = rdata.get(key, None).get('raw_data_filename', None)
            # fname = "{}{}_RAW_V1.csv".format(self.datadir, self.ticker
            fpath = self.datadir + fname
            print("Reading from: {}".format(fpath))
            self.filename_raw = fpath
            self.df_raw = pd.read_csv(
                self.filename_raw, header=0, index_col="Date", parse_dates=True
            )
    def read_remote(self, until, since='2006-12-13'):
        startdate = since
        enddate = datetime.strptime(until, self.dateformat)
        self.remote_data = pdr.get_data_yahoo([self.ticker], start=startdate, end=enddate)
    
    def save_remote_data(self):
        fpath = ''
        print('saving data to {}'.format(fpath))
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
        
        with open('registration.json', encoding='UTF-8') as f:
            self.registraion = json.load(f)

    def set_model_dir(self, dirname):
        self.modeldir = dirname

    def set_ticker(self, ticker):
        self.ticker = ticker

    def set_forecast_horizon(self, val):
        self.forecast_horizon = val

    def set_rolling_window(self, val):
        self.rolling_window = val

    def read_model_local(self, latest=True):
        rdata = self.registration.get(self.ticker, None).get('model', None)
        if latest:
            key = str(max([int(i) for i in rdata.keys()]))
        else:
            key = "0"
        model_name = rdata.get(key, None).get('model_name', None)
        model_path = self.modeldir + model_name
        
        # model_name = "{}LSTM_{}_RW{}_FH{}_V1.h5".format(
        #     self.modeldir, self.ticker, self.rolling_window, self.forecast_horizon
        # )
        
        print("Loading model from {}".format(model_path))
        self.model = load_model(model_path)

    def get_model_list(self):
        """
        Incomplete implemenation
        """
        return
