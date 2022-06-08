import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe, vocab

from custom_dataset import AudioDataset


def create_cluster(dataset, num_clusters, features):
    _dataset = dataset.copy()
    x = _dataset.loc[:, features]

    kmeans = KMeans(num_clusters)
    kmeans.fit(_dataset[features])
    clusters = kmeans.fit_predict(x)

    return clusters


def plot_cluster(dataset, features, label, class_name):
    legend = []
    for name, group in dataset.groupby(label):
        points = group[features].values
        plt.scatter(points[:, 0], points[:, 1])
        legend.append(f'({group.iloc[0][label]}){group.iloc[0][class_name]}')

    plt.legend(legend)
    plt.xlabel('Valance')
    plt.ylabel('Arousal')
    plt.title(f'Valence-Arousal {len(legend)}-clusters')

    plt.show()


def balance_data(dataset, label):
    if len(dataset) == 0:
        return dataset

    balanced = []
    groups = dataset.groupby(label)
    max_group_size = len(max(groups, key=lambda group: len(group[1]))[1])
    for name, group in groups:
        balanced.append(group.append([group.iloc[0]] * (max_group_size - len(group)), ignore_index=True))

    return pd.concat(balanced)


def split_data(dataset, shuffle=False, splits=None, show_result=False):
    if splits is None:
        splits = []

    _dataset = dataset.copy()

    splits = [0] + splits

    num_splits = (np.array(splits) * len(_dataset)).astype(int).cumsum()

    if shuffle:
        idx = np.arange(len(_dataset))
        np.random.shuffle(idx)
        _dataset = _dataset[idx]

    datasets = []
    for i in range(len(num_splits) - 1):
        start, end = num_splits[i], num_splits[i + 1]
        datasets.append(_dataset[start:end])

    datasets = [_dataset[num_splits[-1]:]] + datasets

    if show_result:
        for dataset in datasets:
            plt.scatter(dataset.valence, dataset.arousal)
            plt.xlabel('Valance')
            plt.ylabel('Arousal')
            plt.title(f'Data count: {len(dataset)}')
            plt.show()

    return datasets


def pad_collate(batch, pad_value):
    (specs, lyrics, labels) = zip(*batch)

    padded_lyrics = pad_sequence(lyrics, padding_value=pad_value)

    specs, padded_lyrics, labels = torch.stack(specs), padded_lyrics.T, torch.tensor(labels)

    return specs, padded_lyrics, labels


def get_data_loader(validation_split=0.2,
                    test_split=0.1,
                    batch_size=256,
                    dataset_dir=''):
    df = pd.read_csv(os.path.join(dataset_dir, 'dataset.csv'))

    df_lyrics = pd.read_csv(os.path.join(dataset_dir, 'cache/lyrics-cleaned.csv'))

    assert len(df) == len(df_lyrics)

    # Merge lyrics dataset
    df = df.merge(df_lyrics, on='dzr_sng_id')

    # Clustering (Warning: Class order might change with re-clustering)
    # df['label'] = create_cluster(df, 4, ['valence', 'arousal'])
    # dataset.to_csv(os.path.join(dataset_dir, 'dataset.csv'), index=None)

    # Balance data
    # min_data_size = df.groupby('label').size().values.min()
    # balanced_df = []
    # for name, group in df.groupby('label'):
    #     balanced_df.append(group[:min_data_size])
    # df = pd.concat(balanced_df)

    # classes_name = {0: 'HV-LA',
    #                 1: 'HV-HA',
    #                 2: 'LV-LA',
    #                 3: 'LV-HA'}
    #
    # df['class_name'] = df.label.apply(lambda label: classes_name[label])

    # plot_cluster(df, features=['valence', 'arousal'],
    #              label='label',
    #              class_name='class_name')

    # Add file path
    df['file_path'] = df.dzr_sng_id.apply(lambda song_id: os.path.join(dataset_dir, f'cache/specs/{song_id}.npy'))

    df = shuffle(df)

    df_train, df_val, df_test = split_data(df,
                                           splits=[validation_split, test_split],
                                           show_result=False)

    print('Training data:', len(df_train))
    print('Validation data:', len(df_val))
    print('Test data:', len(df_test))

    # Build vocab from training set for lyrics
    # vocab = torchtext.vocab.build_vocab_from_iterator([df_train.lyrics.str.cat().split(' ')],
    #                                                   specials=["<unk>", "<pad>"])
    # vocab.set_default_index(vocab['<unk>'])
    # pad_value = vocab["<pad>"]

    glove = GloVe(name="6B", dim=200, max_vectors=10000)

    _vocab = vocab(glove.stoi)
    _vocab.insert_token('<unk>', 0)
    _vocab.insert_token('<pad>', 1)
    _vocab.set_default_index(0)

    # TEXT = Field(
    #     lower=True, include_lengths=False, batch_first=True
    # )
    # TEXT.build_vocab(
    #     [row.lyrics.split(' ') for i, row in df.iterrows()],
    #     vectors=glove,
    #     max_size=50_000,
    #     specials=['<unk>', '<pad>']
    # )

    pad_value = _vocab['<pad>']

    val_data = AudioDataset(df=df_val, vocab=_vocab)

    train_data = AudioDataset(df=df_train, vocab=_vocab)

    test_data = AudioDataset(df=df_test, vocab=_vocab)

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=lambda batch: pad_collate(batch, pad_value))

    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=lambda batch: pad_collate(batch, pad_value))

    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=lambda batch: pad_collate(batch, pad_value))

    return train_loader, val_loader, test_loader, _vocab, glove.vectors
