import itertools
import re
from collections import OrderedDict

import numpy as np
import ray
import torch
import torchaudio
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

ray.init(ignore_reinit_error=True, num_cpus=10)


class AudioUtil:
    sample_rate = 22050
    duration = 30
    pitch_shift = 0.3
    fft = 1024
    n_mels = 64
    hop_length = None
    top_db = 80
    max_mask_pct = 0.1
    num_samples = duration * sample_rate

    @staticmethod
    def load_audio(path, mono=True):
        signal, sr = torchaudio.load(path)

        # stereo to mono
        if mono:
            signal = signal.mean(0)

        return signal, sr

    @staticmethod
    def wav_to_melspectrogram(signal, sample_rate):

        # all signal must have same sample rate
        assert AudioUtil.sample_rate == sample_rate

        # pad shorter length samples
        if signal.shape[0] < AudioUtil.num_samples:
            signal = torch.from_numpy(
                np.pad(signal, int(np.ceil((AudioUtil.num_samples - signal.shape[0]) / 2)), mode='reflect'))

        signal = signal[:AudioUtil.num_samples]

        # time shift
        sig_len = len(signal)
        shift_amt = int(np.random.random() * AudioUtil.pitch_shift * sig_len)
        signal = signal.roll(shift_amt)

        # convert to spectrogram
        spec = torchaudio.transforms.MelSpectrogram(sample_rate=AudioUtil.sample_rate,
                                                    n_fft=AudioUtil.fft,
                                                    hop_length=AudioUtil.hop_length,
                                                    n_mels=AudioUtil.n_mels)(signal)
        spec = torchaudio.transforms.AmplitudeToDB(top_db=AudioUtil.top_db)(spec)

        n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = AudioUtil.max_mask_pct * n_mels
        time_mask_param = AudioUtil.max_mask_pct * n_steps

        aug_spec = aug_spec[None, :]
        aug_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
        aug_spec = torchaudio.transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        # Add 1-channel
        # spec = spec[None, :]

        return aug_spec

    @staticmethod
    @ray.remote
    def save_batch_spectrogram(df):
        print(f'Processing batch:{df.iloc[0].dzr_sng_id}...{df.iloc[-1].dzr_sng_id}')

        specs = []
        for i, row in df.iterrows():
            signal, sr = AudioUtil.load_audio(f'{row.file_path}', mono=True)
            spec = AudioUtil.wav_to_melspectrogram(signal=signal, sample_rate=sr)

            specs.append(spec)
            np.save(f'{row.save_file_path}', spec)

        return specs

    @staticmethod
    def save_spectrogram(df):
        start = 0
        end = len(df)
        batch_size = 500

        job_ref = []
        for i in range(start, end, batch_size):
            job_ref.append(AudioUtil.save_batch_spectrogram.remote(df=df[i:min(end, i + batch_size)]))

        ray.get(job_ref)


class LyricsUtil:
    @staticmethod
    def clean_lyrics(phrase):
        # Remove newline and carriage return
        phrase = re.sub("\r\n", " ", phrase).lower()

        # Expand contractions
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        phrase = re.sub(r"wanna", "want to", phrase)
        phrase = re.sub(r"gonna", "going to", phrase)
        phrase = re.sub(r"in'", "ing", phrase)

        # Remove non-alphabetic character
        phrase = re.sub(r"([;.,!?'])", " ", phrase)
        phrase = re.sub("[^a-z]+", " ", phrase)

        # Remove extra space
        phrase = re.sub(" +", " ", phrase)
        phrase = phrase.strip()

        # Remove stop words like a, and, the
        phrase = [word for word in phrase.split(' ') if word not in stopwords.words('english')]

        # Lemmanize words, eg plural to singular form, past to present form
        lemmatizer = WordNetLemmatizer()
        phrase = [lemmatizer.lemmatize(word) for word in phrase]

        # Remove duplicate but preserve order
        phrase = ' '.join(OrderedDict.fromkeys(phrase))

        return phrase

    @staticmethod
    @ray.remote
    def clean_batch_lyrics(df, batch_id):
        print(f'Processing batch:{df.iloc[0].dzr_sng_id}...{df.iloc[-1].dzr_sng_id}')

        lyrics = []
        for i, row in df.iterrows():
            lyric = LyricsUtil.clean_lyrics(row.lyrics)
            lyrics.append(lyric)

        return lyrics

    @staticmethod
    def get_cleaned_lyrics(df):
        start = 0
        end = len(df)
        batch_size = 500

        job_ref = []
        for i in range(start, end, batch_size):
            job_ref.append(LyricsUtil.clean_batch_lyrics.remote(df=df[i:min(end, i + batch_size)], batch_id=i))

        lyrics_list = ray.get(job_ref)

        lyrics = list(itertools.chain.from_iterable(lyrics_list))

        return lyrics
