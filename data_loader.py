import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from torch.utils.data import DataLoader

from custom_dataset import AudioDataset


def create_cluster(dataset, num_clusters=None, show_cluster=False):
    _dataset = dataset.copy()
    x = _dataset.loc[:, ['valence', 'arousal']]

    if num_clusters:
        kmeans = KMeans(num_clusters)
        kmeans.fit(_dataset[['valence', 'arousal']])
        clusters = kmeans.fit_predict(x)
        _dataset['cluster'] = clusters

    if not show_cluster:
        return clusters

    fig, ax = plt.subplots(1, figsize=(8, 8))

    plt.scatter(_dataset.valence, _dataset.arousal, c=_dataset.cluster)

    for i in _dataset.cluster.unique():
        points = _dataset[_dataset.cluster == i][['valence', 'arousal']].values
        hull = ConvexHull(points)

        x_hull = np.append(points[hull.vertices, 0],
                           points[hull.vertices, 0][0])
        y_hull = np.append(points[hull.vertices, 1],
                           points[hull.vertices, 1][0])

        dist = np.sqrt((x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2)
        dist_along = np.concatenate(([0], dist.cumsum()))
        spline, _ = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0)
        interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
        interp_x, interp_y = interpolate.splev(interp_d, spline)

        plt.plot(interp_x, interp_y, '-', c="r")
        plt.xlabel('Valance')
        plt.ylabel('Arousal')
        plt.title(f'Valence-Arousal {num_clusters}-clusters')

    plt.show()

    return clusters


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
                    num_classes=4,
                    batch_size=256,
                    dataset_dir=''):
    dataset = pd.read_csv(os.path.join(dataset_dir, 'dataset.csv'))
    dataset = shuffle(dataset)

    dataset['label'] = create_cluster(dataset,
                                      num_clusters=num_classes,
                                      show_cluster=False)

    train_split, val_split, test_split = split_data(dataset,
                                                    splits=[validation_split, test_split],
                                                    show_result=False)

    train_data = AudioDataset(df=train_split,
                              feature_label='dzr_sng_id',
                              target_label='label',
                              audio_directory=os.path.join(dataset_dir, 'previews/wav'),
                              preload=True,
                              transform=None)

    val_data = AudioDataset(df=val_split,
                            feature_label='dzr_sng_id',
                            target_label='label',
                            audio_directory=os.path.join(dataset_dir, 'previews/wav'),
                            preload=True,
                            transform=None)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, None
