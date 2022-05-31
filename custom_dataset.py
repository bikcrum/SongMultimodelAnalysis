import itertools
import os
import time

import numpy as np
import ray
import torch
import torchaudio
from torch.utils.data import Dataset

ray.init(ignore_reinit_error=True, num_cpus=10)
os.environ['RAY_verbose_spill_logs'] = '0'


class AudioUtil:
    sample_rate = 22050
    duration = 30
    pitch_shift = 0.3
    fft = 1024
    n_mels = 64
    hop_length = None
    top_db = 80
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

        # convert to spectrogram
        spec = torchaudio.transforms.MelSpectrogram(sample_rate=AudioUtil.sample_rate,
                                                    n_fft=AudioUtil.fft,
                                                    hop_length=AudioUtil.hop_length,
                                                    n_mels=AudioUtil.n_mels)(signal)
        spec = torchaudio.transforms.AmplitudeToDB(top_db=AudioUtil.top_db)(spec)

        # Add 1-channel
        spec = spec[None, :]

        return spec


@ray.remote
def save_spectrogram(df, feature_label, audio_directory, save_directory, batch_id):
    print(f'Storing data:{df.iloc[0][feature_label]}...{df.iloc[-1][feature_label]}')

    specs = []
    for i, row in df.iterrows():
        signal, sr = AudioUtil.load_audio(
            os.path.join(audio_directory, f'{row[feature_label]}.wav'),
            mono=True)
        spec = AudioUtil.wav_to_melspectrogram(signal=signal, sample_rate=sr)

        specs.append(spec)
        np.save(os.path.join(save_directory, f'{int(row[feature_label])}.npy'), spec)

    return specs


def build_cache(df, feature_label, audio_directory, save_directory, replace=False):
    os.makedirs(save_directory, exist_ok=True)

    _df = df.copy()

    available_data = list(map(lambda file: int(file.split('.')[0]), os.listdir(save_directory)))

    # if replace is false then cache remaining data only
    if not replace:
        # filter remaining data to be cached
        rids = set(_df[feature_label]) - set(available_data)
        _df = _df[_df[feature_label].isin(rids)]

    print(f'{len(available_data)} data available in the disk. {len(_df)} data is going to be cached.')

    # Cache remaining data if replace is false otherwise cache everything
    start = 0
    end = len(_df)
    batch_size = 500

    job_ref = []
    for i in range(start, end, batch_size):
        job_ref.append(save_spectrogram.remote(df=_df[i:min(end, i + batch_size)],
                                               feature_label=feature_label,
                                               audio_directory=audio_directory,
                                               save_directory=save_directory,
                                               batch_id=i))

    specs = ray.get(job_ref)

    # When everything was stored at this time, no need to load from disk
    if len(_df) == len(df):
        return np.array(list(itertools.chain.from_iterable(specs)))

    # If at least one data was cache load everything from disk
    specs = []
    for i, row in df.iterrows():
        spec = np.load(os.path.join(save_directory, f'{row[feature_label]}.npy'))
        specs.append(spec)

    return np.array(specs)


class AudioDataset(Dataset):

    def __init__(self,
                 df,
                 feature_label='audio_id',
                 target_label='label',
                 audio_directory='',
                 preload=True,
                 transform=None):
        super().__init__()

        self.df = df.copy()
        self.feature_label = feature_label
        self.target_label = target_label
        self.transform = transform
        self.preload = preload
        self.audio_directory = audio_directory

        if self.preload:
            save_directory = os.path.join(os.path.dirname(audio_directory), 'melspec')
            t = time.time()
            self.data = build_cache(df,
                                    feature_label,
                                    audio_directory,
                                    save_directory,
                                    replace=False)
            print('Data prepared in:', time.time() - t)

            assert len(self.data) == len(df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        if self.preload:
            spec = self.data[idx]
            spec = torch.Tensor(spec)
        else:
            signal, sr = AudioUtil.load_audio(
                os.path.join(self.audio_directory, f'{self.df.iloc[idx][self.feature_label]}.wav'),
                mono=True)

            spec = AudioUtil.wav_to_melspectrogram(signal=signal, sample_rate=sr)

        if self.transform:
            spec = self.transform(spec)

        class_label = self.df.iloc[idx][self.target_label]

        return spec, class_label
