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
from keras.models import Sequential, Model


class CustomModel:
    def __init__(self) -> None:
        self.model = Sequential()

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
