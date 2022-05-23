import tensorflow as tf

keras = tf.keras
import pandas as pd
import pickle
import logging
import numpy as np
import datetime

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

dataset = pd.read_csv('dataset/dataset.csv')

logging.info('Reading data from disk')

waveforms = []
for song_id in dataset.dzr_sng_id:
    with open(f'dataset/previews/melspectrogram/{song_id}.mel', 'rb') as r:
        waveforms.append(pickle.load(r))

waveforms = np.array(waveforms)
mean = (waveforms - waveforms.mean()) / waveforms.std()


input_size = (waveforms.shape[1], waveforms.shape[2])
sample_rate = dataset.iloc[0].sample_rate

logging.info('Data load done.')

_FFT_SIZE = 1024
_HOP_SIZE = 512
_N_MEL_BINS = 64
_N_SPECTROGRAM_BINS = (_FFT_SIZE // 2) + 1
_F_MIN = 0.0
_F_MAX = sample_rate / 2


def AudioModel(input_size, n_mels):
    x = keras.layers.Input(shape=input_size, name='input', dtype=tf.float32)
    y = tf.expand_dims(x, 3)

    # effectively 1D convolution, since kernel spans entire frequency-axis
    y = keras.layers.Conv2D(32, (3, n_mels), activation='relu')(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.MaxPool2D((1, y.shape[2]))(y)

    y = keras.layers.Conv2D(32, (3, 1), activation='relu')(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.MaxPool2D(pool_size=(2, 1))(y)

    y = keras.layers.Flatten()(y)
    y = keras.layers.Dense(64, activation='relu')(y)

    y = keras.layers.Dense(units=2)(y)

    return keras.Model(inputs=x, outputs=y)


model = AudioModel(input_size=input_size, n_mels=_N_MEL_BINS)

model.summary()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003),
              loss='mean_absolute_error', metrics=['accuracy'])

checkpoint_filepath = 'tf-checkpoints/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

log_dir = "tf-logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# model.load_weights(checkpoint_filepath)

model.fit(waveforms,
          dataset[['arousal', 'valence']],
          callbacks=[model_checkpoint_callback, tensorboard_callback],
          validation_split=0.3,
          epochs=100)
