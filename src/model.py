import os
from src.utils import Timer
from keras.layers import (
    Dense,
    Input,
    Activation,
    Dropout,
    RepeatVector,
    TimeDistributed,
    Flatten,
)
from keras.layers import LSTM
from keras.models import Sequential, Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime as dt

class CustomModel:
    def __init__(self) -> None:
        self.model = Sequential()
        self.config = None
        self.log_dict = {
                "rolling_window": None,
                "forecast_horizon": None,
                "data_filename": None,
                "data_version": None,
                "model_name": None,
                "training_history": None,
                "test_data": None,
                "epochs": None,
                "batch_size": None,
                "drop_rate": None,
                "n_units": None,
                "batch_size": None,
                "date_of_update": None,
                "date_of_entry": None
            }
        
    def set_config(self, config):
        self.config = config
        # self.modeldir = self.config.get_dir_model()
        # self.training_history_dir = self.config.get_dir_training_history()
        self.training_config = self.config.get_config_training()
        self.epochs = self.training_config.get('epochs', None)
        self.batch_size = self.training_config.get('batch_size', None)
        self.model_path = self.config.get_path_to_save_model()
        self.log = self.config.get_log()
        self.log_ticker = self.config.get_log_by_ticker()
        
    def build_model(self, input_n, output_n, drop_rate, latent_n, feature_n):
        """
        input_n: the length of the input sequence
        output_n: the length of the predicted sequence
        feature_n: how many features we have in the model
        latent_n: the size of the hidden units.
        """

        print("input_n", input_n)
        print("output_n", output_n)
        print("latent_n", latent_n)
        print("feature_n", feature_n)

        # =============================================================================
        # Bidirectional LSTM
        # =============================================================================

        #    3/26/2020

        encoder_inputs = Input(shape=(input_n, feature_n))

        # unidirectional LSTM layer
        encoder = LSTM(latent_n, return_state=True, activation="relu")

        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = RepeatVector(output_n)(encoder_outputs)
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(
            latent_n, return_sequences=True, return_state=True, activation="relu"
        )
        decoder_outputs, _, _ = decoder_lstm(
            decoder_inputs, initial_state=encoder_states
        )

        #    decoder_outputs_1 = Dense(output_n)(decoder_outputs)

        inter_output = Dropout(drop_rate)(decoder_outputs)

        decoder_outputs_2 = Dense(1)(inter_output)

        #    final_output = Dropout(drop_rate)(decoder_outputs)

        model = Model(encoder_inputs, decoder_outputs_2)

        #    start = time.time()
        model.compile(loss="mean_squared_error", optimizer="adam")

        #    print("Compilation Time:" + str(time.time()-start) )
        print(model.summary())

        self.model = model

    def train(self, X_train, y_train, X_test, y_test):
        timer = Timer()
        timer.start()
        print("[Model] Training Started")
        print("[Model] %s epochs, %s batch size" % (self.epochs, self.batch_size))
        
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=2),
            ModelCheckpoint(
                filepath=self.model_path, monitor="val_loss", save_best_only=True
            ),
        ]
        
        history = self.model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            validation_data=(X_test, y_test),
        )
        self.model.save(self.model_path)
        
        # save history here
        self.config.log_training_history(history.history)
        # save log
        self.create_log_model()
        
        print("[Model] Training Completed. Model saved as %s" % self.model_path)
        timer.stop()
    
    def create_log_model(self):
        self.log_dict['rolling_window'] = self.config.get_config_data().get('rolling_window', None)
        self.log_dict['forecast_horizon'] = self.config.get_config_data().get('forecast_horizon', None)
        self.log_dict['data_filename'] = self.config.get_config_data().get('data_filename', None)
        self.log_dict['data_version'] = self.config.get_config_data().get('data_version', None)
        self.log_dict['test_data'] = self.config.get_config_data().get('test_data', None)
        
        # self.log_dict['model_name'] = self.log_dict['raw_data_filename'].replace('RAW', 'TA')
        # self.log_dict['training_history'] = self.log_dict['raw_data_filename'].replace('RAW', 'TA')
        
        self.log_dict['epochs'] = self.epochs
        self.log_dict['batch_size'] = self.batch_size
        
        # self.log_dict['drop_rate'] = self.config.get_config_model().get()
        # self.log_dict['n_units'] = self.config.get_config_model().get()
        
        # self.log_dict['date_of_update'] = self.log_dict['raw_data_filename'].replace('RAW', 'TA')
        # self.log_dict['date_of_entry'] = self.log_dict['raw_data_filename'].replace('RAW', 'TA')
        
        latest_id = int(self.config.get_latest_id_by_ticker(ticker=self.config.ticker, tickerchild='model'))
        self.log[self.config.ticker]['model'][str(latest_id + 1)] = self.log_dict
        
        self.config.save_log(self.log)
        
        print('reading log after writing')
        self.log = self.config.get_log()
        self.log_ticker = self.config.get_log_by_ticker()
        