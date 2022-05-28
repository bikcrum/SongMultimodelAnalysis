import numpy as np
import tensorflow as tf

keras = tf.keras
import pandas as pd
import librosa as lr
import logging
import pickle
from tqdm import tqdm

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def convert(dataset, batch_id):
    logging.info('Reading data from disk')
    signals, _ = zip(*map(lr.load, 'dataset/previews/wav/' + dataset.dzr_sng_id.astype(str) + '.wav'))

    max_sample_size = dataset.sample_size.max()

    # add silence for song smaller than maximum length
    samples = np.array(
        list(map(lambda isignal: tf.concat(
            [isignal[1], tf.zeros(max_sample_size - dataset.iloc[isignal[0]].sample_size)],
            axis=0),
                 enumerate(signals))))
    logging.info('Data load done.')

    _FFT_SIZE = 1024
    _HOP_SIZE = 512
    _N_MEL_BINS = 64
    _N_SPECTROGRAM_BINS = (_FFT_SIZE // 2) + 1
    _F_MIN = 0.0
    _F_MAX = dataset.iloc[0].sample_rate / 2
    _SAMPLE_RATE = dataset.iloc[0].sample_rate

    mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=_N_MEL_BINS,
        num_spectrogram_bins=_N_SPECTROGRAM_BINS,
        sample_rate=_SAMPLE_RATE,
        lower_edge_hertz=_F_MIN,
        upper_edge_hertz=_F_MAX)

    spectrograms = tf.signal.stft(samples,
                                  frame_length=_FFT_SIZE,
                                  frame_step=_HOP_SIZE)

    magnitude_spectrograms = tf.abs(spectrograms)

    mel_power_spectrograms = tf.matmul(tf.square(magnitude_spectrograms),
                                       mel_filterbank)

    def power_to_db(S, amin=1e-16, top_db=80.0):
        def _tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        ref = tf.reduce_max(S)

        log_spec = 10.0 * _tf_log10(tf.maximum(amin, S))
        log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref))

        log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

        return log_spec

    log_magnitude_mel_spectrograms = power_to_db(mel_power_spectrograms)

    tq = tqdm(range(len(dataset)))
    for i in tq:
        tq.set_description(f'Converting:{batch_id + i}')
        with open(f'dataset/previews/melspectrogram/{dataset.iloc[i].dzr_sng_id}.mel', 'wb') as w:
            pickle.dump(log_magnitude_mel_spectrograms[i], w)


if __name__ == '__main__':
    dataset = pd.read_csv('dataset/dataset.csv')
    start = 0
    end = len(dataset)
    batch_size = 100
    for i in range(start, end, batch_size):
        convert(dataset[i:min(end, i + batch_size)], i)
