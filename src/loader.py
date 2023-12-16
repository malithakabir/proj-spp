import os
import numpy as np
import pandas as pd
import re
import json
from keras.models import load_model

# import talib as tb
from sklearn.preprocessing import StandardScaler
from pandas_datareader import data as pdr
from datetime import datetime
import yfinance as yf

yf.pdr_override()


class ConfigLoader:
    def __init__(self) -> None:
        self.config = None
        self.config_filename = None
        self.config_dir = None
        self.config_filepath = None
        self.log = None
        self.basename = None
    def set_ticker(self, ticker):
        print('setting up ticker: ' + ticker)
        self.ticker = ticker
        
    def set_config_dir(self, dirname="config/"):
        self.config_dir = dirname

    def set_config_filename(self, filename="config.json"):
        self.config_filename = filename

    def set_config_filepath(self, dirname=None, filename=None):
        if dirname == None or filename == None:
        # if self.config_dir == None or self.config_filename == None:
            self.set_config_dir()
            self.set_config_filename()
        self.config_filepath = self.config_dir + self.config_filename
        print('config file path ' + self.config_filepath)

    def read_local_config(self):
        if self.config_filepath == None:
            self.set_config_filepath()
        with open(self.config_filepath, encoding="UTF-8") as config_file:
            print('reading local config file')
            self.config = json.load(config_file)
            
    def get_ticker(self):
        return self.ticker
    def get_dir_data(self):
        return self.config.get("dir", None).get("data")

    def get_dir_model(self):
        return self.config.get("dir", None).get("model")

    def get_dir_log(self):
        return self.config.get("dir", None).get("log")

    def get_dir_training_history(self):
        return self.config.get("dir", None).get("training_history")

    def get_config_data(self):
        return self.config.get("data", None)

    def get_config_training(self):
        return self.config.get("training", None)

    def get_config_model(self):
        return self.config.get("model", None)

    def get_dateformat(self):
        return self.config.get("date_format", None)

    def get_log_filename(self):
        return self.config.get("filename", None).get("logfile", None)

    def get_log_filepath(self):
        return self.get_dir_log() + self.get_log_filename()

    def read_local_log(self):
        log_filepath = self.get_log_filepath()
        with open(log_filepath, encoding="UTF-8") as log_file:
            self.log = json.load(log_file)
    def get_log(self):
        if self.log is None:
            self.read_local_log()
        return self.log
        
    def get_log_by_ticker(self):
        return self.log.get(self.ticker, None)
        
    def get_latest_id_by_ticker(self, ticker, tickerchild):
        data_or_model_dict = self.log.get(ticker, None).get(tickerchild, None)
        latest_data_or_model_id = max([int(idx) for idx in data_or_model_dict.keys()])
        return str(latest_data_or_model_id)

    def generate_basename_from_config(self, model_type="LSTM"):
        dataconfig = self.config.get("data", None)
        ticker = dataconfig.get("data_filename", None).split("_")[0]
        rw = dataconfig.get("rolling_window", None)
        fh = dataconfig.get("forecast_horizon", None)
        version = int(self.get_latest_id_by_ticker(ticker=ticker, tickerchild="model")) + 1
        self.basename = "{}_{}_RW{}_FH{}_V{}".format(
            ticker, model_type, rw, fh, version
        )

    def log_training_history(self, history, dirname=None):
        if dirname is None:
            dirname = self.get_dir_training_history()
        fpath = dirname + self.basename + ".json"
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)

    def get_path_to_save_model(self):
        if self.basename is None:
            self.generate_basename_from_config()
        model_path = self.get_dir_model() + self.basename + ".h5"
        return model_path

    def get_path_to_save_raw_data(self, ticker, create_version=True):
        current = int(self.get_latest_id_by_ticker(ticker=ticker, tickerchild="data"))
        version = current if not create_version else current + 1
        data_path_raw = self.get_dir_data() + "{}_RAW_V{}.csv".format(ticker, version)
        return data_path_raw

    def get_path_to_save_technical_analysis(self, ticker, create_version=False):
        current = int(self.get_latest_id_by_ticker(ticker=ticker, tickerchild="data"))
        version = current if not create_version else current + 1
        data_path_ta = self.get_dir_data() + "{}_TA_V{}.csv".format(ticker, version)
        return data_path_ta
    def save_log(self, log):
        with open(self.get_log_filepath(), "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=4)
        self.get_log()
        
        
class DataLoader:
    def __init__(self, ticker=None, config=None):
        # print("")

        # if ticker is not None:
        #     self.ticker = ticker.upper()
        # else:
        #     print("Ticker is missing")

        # if config is not None:
        #     self.set_config(config)
        # else:
        #     print("No config found")

        self.isTechnical = None
        self.filename_processed = None
        self.df_processed = None
        self.filename_raw = None
        self.df_raw = None
        # self.remote_data = None
        self.log_dict = {
                "raw_data_filename": None,
                "raw_data_date_of_latest_data_point": None,
                "raw_data_date_of_first_data_point": None,
                "technical_analysis_filename": None,
                "date_of_entry": None
            }

    def set_config(self, config):
        self.config = config
        self.dateformat = self.config.get_dateformat()
        self.datadir = self.config.get_dir_data()
        self.ticker = self.config.get_ticker()
        self.log = self.config.get_log()
        self.log_ticker = self.config.get_log_by_ticker()
        
    # def set_datadir(self, dirname):
    #     self.datadir = dirname

    def set_ticker(self, ticker):
        self.ticker = ticker

    def read_local(self, isTechnical=True):
        # reading the latest data
        self.isTechnical = isTechnical
        latest_version = self.config.get_latest_id_by_ticker(
            ticker=self.ticker, tickerchild="data"
        )
        key = str(latest_version)

        if isTechnical:
            fname = (
                self.log_ticker.get("data", None)
                .get(key, None)
                .get("technical_analysis_filename", None)
            )
            # fname = "{}{}_TA_V1.csv".format(self.datadir, self.ticker)
            fpath = self.datadir + fname
            print("Reading from: {}".format(fpath))
            self.filename_processed = fpath
            self.df_processed = pd.read_csv(
                self.filename_processed, parse_dates=True, index_col=0
            )
        else:
            fname = (
                self.log_ticker.get("data", None)
                .get(key, None)
                .get("raw_data_filename", None)
            )
            # fname = "{}{}_RAW_V1.csv".format(self.datadir, self.ticker
            fpath = self.datadir + fname
            print("Reading from: {}".format(fpath))
            self.filename_raw = fpath
            self.df_raw = pd.read_csv(
                self.filename_raw, header=0, index_col="Date", parse_dates=True
            )

    def read_remote(self, until, since="2006-12-13"):
        startdate = since
        enddate = datetime.strptime(until, self.dateformat)
        self.raw_data = pdr.get_data_yahoo(
            [self.ticker], start=startdate, end=enddate
        )
    
    def create_log_raw_data(self):
        latest_id = int(self.config.get_latest_id_by_ticker(ticker=self.ticker, tickerchild='data'))
        self.log[self.ticker]['data'][str(latest_id + 1)] = self.log_dict
        
        self.config.save_log(self.log)
        
        print('reading log after writing')
        self.log = self.config.get_log()
        self.log_ticker = self.config.get_log_by_ticker()
        
    def save_raw_data(self):
        # datadir = self.config.get_dir_data()
        # latest_version = self.config.get_latest_id_by_ticker(ticker=self.ticker, type='data')
        fpath = self.config.get_path_to_save_raw_data(self.ticker, create_version=True)
        self.raw_data.to_csv(fpath, sep=',')
        self.log_dict['raw_data_filename'] = fpath.split('/')[-1]
        self.log_dict['technical_analysis_filename'] = self.log_dict['raw_data_filename'].replace('RAW', 'TA')
        print("raw data saved to : {}".format(fpath))
        self.create_log_raw_data()

    def set_df_processed(self, df_processed):
        self.df_processed = df_processed
        
    def save_technical_analysis(self):
        # not logging here
        fpath = self.config.get_path_to_save_technical_analysis(self.ticker, create_version=False)
        self.df_processed.to_csv(fpath, sep=',')
        print("technical analysis data saved to : {}".format(fpath))


class ModelLoader:
    def __init__(self, config=None) -> None:
        self.ticker = None
        self.modeldir = None
        self.forecast_horizon = None
        self.rolling_window = None
        self.model = None
        self.config = None
        self.dateformat = None
        self.model_version = None
        self.model_checkpoint_filepath = None
        self.model_history_filepath = None
        self.training_config = None
        
    def set_config(self, config):
        self.config = config
        self.ticker = self.config.ticker
        self.dateformat = self.config.get_dateformat()
        self.modeldir = self.config.get_dir_model()
        self.training_history_dir = self.config.get_dir_training_history()
        self.forcast_horizon = self.config.get_config_data().get("forecast_horizon", None)
        self.rolling_window = self.config.get_config_data().get("rolling_window", None)
        self.test_data = self.config.get_config_data().get("test_data", None)
        
    # def set_model_dir(self, dirname):
    #     self.modeldir = dirname

    # def set_ticker(self, ticker):
    #     self.ticker = ticker

    # def set_forecast_horizon(self, val):
    #     self.forecast_horizon = val

    # def set_rolling_window(self, val):
    #     self.rolling_window = val

    def read_model_local(self):
        key = self.config.get_latest_id_by_ticker(self.ticker, type="model")
        model_checkpoint_filename = (
                self.log_ticker.get("model", None)
                .get(key, None)
                .get("model_name", None)
            )
        self.model_checkpoint_filepath = self.modeldir + model_checkpoint_filename
        
        model_history_filename = (
                self.log_ticker.get("model", None)
                .get(key, None)
                .get("training_history", None)
            )
        self.model_history_filepath = self.training_history_dir + model_history_filename
        
        # model_name = "{}LSTM_{}_RW{}_FH{}_V1.h5".format(
        #     self.modeldir, self.ticker, self.rolling_window, self.forecast_horizon
        # )

        print("Loading model from {}".format(self.model_checkpoint_filepath))
        self.model = load_model(self.model_checkpoint_filepath)
        
