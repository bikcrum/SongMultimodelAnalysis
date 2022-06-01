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

        # hull = ConvexHull(points)
        #
        # x_hull = np.append(points[hull.vertices, 0],
        #                    points[hull.vertices, 0][0])
        # y_hull = np.append(points[hull.vertices, 1],
        #                    points[hull.vertices, 1][0])
        #
        # dist = np.sqrt((x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2)
        # dist_along = np.concatenate(([0], dist.cumsum()))
        # spline, _ = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0)
        # interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
        # interp_x, interp_y = interpolate.splev(interp_d, spline)
        #
        # plt.plot(interp_x, interp_y, '-')

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
                    num_classes=4,
                    batch_size=256,
                    classes_name=None,
                    dataset_dir=''):
    dataset = pd.read_csv(os.path.join(dataset_dir, 'dataset.csv'))
    target_label = 'label'

    # dataset['label'] = create_cluster(dataset, 4, ['valence', 'arousal'])
    # dataset.to_csv(os.path.join(dataset_dir, 'dataset.csv'), index=None)

    assert len(classes_name) == num_classes

    plot_cluster(dataset, ['valence', 'arousal'], target_label, classes_name)

    dataset = shuffle(dataset)

    train_split, val_split, test_split = split_data(dataset,
                                                    splits=[validation_split, test_split],
                                                    show_result=False)

    # train_split = balance_data(train_split, target_label)
    # val_split = balance_data(val_split, target_label)
    # test_split = balance_data(test_split, target_label)

    train_data = AudioDataset(df=train_split,
                              feature_label='dzr_sng_id',
                              target_label=target_label,
                              audio_directory=os.path.join(dataset_dir, 'previews/wav'),
                              preload=True,
                              transform=None)

    val_data = AudioDataset(df=val_split,
                            feature_label='dzr_sng_id',
                            target_label=target_label,
                            audio_directory=os.path.join(dataset_dir, 'previews/wav'),
                            preload=True,
                            transform=None)

    test_data = AudioDataset(df=test_split,
                             feature_label='dzr_sng_id',
                             target_label=target_label,
                             audio_directory=os.path.join(dataset_dir, 'previews/wav'),
                             preload=True,
                             transform=None)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_data
