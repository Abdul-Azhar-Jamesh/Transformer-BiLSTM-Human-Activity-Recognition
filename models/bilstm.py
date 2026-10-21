"""Stacked BiLSTM for human activity recognition. """
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout
L2_WEIGHT = 1e-4

def build_bilstm(input_shape=(128, 9), num_classes=6):
    """
        Baseline Bidirectional LSTM.
        """
    inputs = Input(shape=input_shape, name='sensor_input')
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(inputs)
    x = Bidirectional(LSTM(64, dropout=0.2))(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(L2_WEIGHT))(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax', name='activity_output')(x)
    model = Model(inputs, outputs, name='Baseline_BiLSTM')
    return model
