import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from torch.utils.data import DataLoader

from custom_dataset import AudioDataset


def create_cluster(dataset, num_clusters, features):
    _dataset = dataset.copy()
    x = _dataset.loc[:, features]

    kmeans = KMeans(num_clusters)
    kmeans.fit(_dataset[features])
    clusters = kmeans.fit_predict(x)

    return clusters


def plot_cluster(dataset, features, class_label, class_names):
    legend = []
    for name, group in dataset.groupby(class_label):
        points = group[features].values
        plt.scatter(points[:, 0], points[:, 1])
        legend.append(class_names[group.iloc[0][class_label]])

    plt.legend(legend)
    plt.xlabel('Valance')
    plt.ylabel('Arousal')
    plt.title(f'Valence-Arousal {len(class_names)}-clusters')

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


def get_data_loader(validation_split=0.2,
                    test_split=0.1,
                    batch_size=256,
                    classes_name=None,
                    dataset_dir=''):
    df = pd.read_csv(os.path.join(dataset_dir, 'dataset.csv'))

    # Merge lyrics dataset
    df_lyrics = pd.read_csv(os.path.join(dataset_dir, 'cache/lyrics-cleaned.csv'))

    assert len(df) == len(df_lyrics)

    df = df.merge(df_lyrics, on='dzr_sng_id')[:1000]

    # Add file path
    df['file_path'] = df.dzr_sng_id.apply(lambda song_id: os.path.join(dataset_dir, f'cache/specs/{song_id}.npy'))

    # dataset['label'] = create_cluster(dataset, 4, ['valence', 'arousal'])
    # dataset.to_csv(os.path.join(dataset_dir, 'dataset.csv'), index=None)

    # plot_cluster(df, features=['valence', 'arousal'],
    #              class_label='label',
    #              class_names=classes_name)

    df = shuffle(df)

    df_train, df_val, df_test = split_data(df,
                                           splits=[validation_split, test_split],
                                           show_result=False)
    train_data = AudioDataset(df=df_train)

    val_data = AudioDataset(df=df_val)

    test_data = AudioDataset(df=df_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
