import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


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


def split_data(dataset, shuffle=False, splits=None):
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

    return [_dataset[num_splits[-1]:]] + datasets


# Preloads all data
class DeezerMusicDataset(Dataset):
    def __init__(self, dataset, target_label='cluster', workdir='', transform=None):
        self.data = []
        for song_id in dataset.dzr_sng_id:
            self.data.append(np.load(os.path.join(workdir, f'dataset/previews/melspectrogram/{song_id}.npy')))

        assert len(dataset) == len(self.data), "Some audio couldn't be loaded"

        bad_shape_data = list(filter(lambda x: x[1].shape != self.data[0].shape, enumerate(self.data)))

        if len(bad_shape_data) > 0:
            bad_shape_data_index, _ = zip(*bad_shape_data)
            raise Exception('Some audio data are corrupted', dataset.iloc[np.array(bad_shape_data_index)].dzr_sng_id)

        self.data = np.array(self.data)
        self.label = dataset[[target_label]].squeeze().values

        self.transform = transform

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        x = Tensor(self.data[idx])

        if self.transform is not None:
            x = self.transform(x)

        y = self.label[idx]

        return x, y


# Loads data during training when needed
class DeezerMusicDatasetOnDemand(Dataset):
    def __init__(self, dataset, target_label='cluster', exclude_missing_file=False, workdir='',
                 transform=None):
        self.workdir = workdir

        self.data = list(filter(
            lambda song_id: os.path.exists(os.path.join(workdir, f'dataset/previews/melspectrogram/{song_id}.npy'))
                            and os.path.isfile(os.path.join(workdir, f'dataset/previews/melspectrogram/{song_id}.npy')),
            dataset.dzr_sng_id))

        if not exclude_missing_file:
            assert len(self.data) == len(
                dataset), "Some files are missing. Use exclude_missing_file=True to exclude them in dataset"

        self.data = np.array(self.data)
        self.labels = dataset[[target_label]].squeeze().values

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = np.load(os.path.join(self.workdir, f'dataset/previews/melspectrogram/{self.data[idx]}.npy'))

        x = Tensor(x)

        if self.transform is not None:
            x = self.transform(x)

        y = self.labels[idx]

        return x, y


# Lazy loads the dataset (Improves initial loading time)
class DeezerMusicDatasetLazyLoad(Dataset):
    def background_batch_fetch(self, i):
        if self.data[i] is not None:
            return

        x = np.load(os.path.join(self.workdir, f'dataset/previews/melspectrogram/{self.data_id[i]}.npy'))

        x = Tensor(x)

        if self.transform is not None:
            x = self.transform(x)

        self.data[i] = x

    def background_fetch(self):
        with Pool(processes=10) as pool:
            pool.map(self.background_batch_fetch, list(range(len(self.data_id))))

    def __init__(self, dataset, target_label='cluster', exclude_missing_file=False, workdir='',
                 transform=None):
        self.workdir = workdir

        self.data_id = list(filter(
            lambda song_id: os.path.exists(os.path.join(workdir, f'dataset/previews/melspectrogram/{song_id}.npy'))
                            and os.path.isfile(os.path.join(workdir, f'dataset/previews/melspectrogram/{song_id}.npy')),
            dataset.dzr_sng_id))

        if not exclude_missing_file:
            assert len(self.data_id) == len(
                dataset), "Some files are missing. Use exclude_missing_file=True to exclude them in dataset"

        self.data_id = np.array(self.data_id)
        self.labels = dataset[[target_label]].squeeze().values

        self.transform = transform

        self.data = [None] * len(self.data_id)

        self.background_fetch()

    def __len__(self):
        return self.data_id.shape[0]

    def __getitem__(self, idx):
        if self.data[idx] is not None:
            return self.data[idx], self.labels[idx]

        x = np.load(os.path.join(self.workdir, f'dataset/previews/melspectrogram/{self.data_id[idx]}.npy'))

        x = Tensor(x)

        if self.transform is not None:
            x = self.transform(x)

        self.data[idx] = x

        y = self.labels[idx]

        return x, y


def get_data_loader(validation_split=0.2,
                    test_split=0.1,
                    num_classes=4,
                    batch_size=256,
                    loader_type='preload',
                    workdir=''):
    assert loader_type == 'preload' or loader_type == 'on_demand' or loader_type == 'lazy_load'

    dataset = pd.read_csv(os.path.join(workdir, 'dataset/dataset.csv'))

    dataset['cluster'] = create_cluster(dataset, num_clusters=num_classes)

    train_split, val_split, test_split = split_data(dataset, splits=[validation_split, test_split])

    if loader_type == 'preload':
        train_data = DeezerMusicDataset(train_split,
                                        target_label='cluster',
                                        workdir=workdir)

        val_data = DeezerMusicDataset(val_split,
                                      target_label='cluster',
                                      workdir=workdir)

        test_data = DeezerMusicDataset(test_split,
                                       target_label='cluster',
                                       workdir=workdir)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    elif loader_type == 'on_demand':
        train_data = DeezerMusicDatasetOnDemand(train_split,
                                                target_label='cluster',
                                                exclude_missing_file=False,
                                                workdir=workdir)

        val_data = DeezerMusicDatasetOnDemand(val_split,
                                              target_label='cluster',
                                              exclude_missing_file=False,
                                              workdir=workdir)

        test_data = DeezerMusicDatasetOnDemand(test_split,
                                               target_label='cluster',
                                               exclude_missing_file=False,
                                               workdir=workdir)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    elif loader_type == 'lazy_load':
        train_data = DeezerMusicDatasetLazyLoad(train_split,
                                                target_label='cluster',
                                                exclude_missing_file=False,
                                                workdir=workdir)

        val_data = DeezerMusicDatasetLazyLoad(val_split,
                                              target_label='cluster',
                                              exclude_missing_file=False,
                                              workdir=workdir)

        test_data = DeezerMusicDatasetLazyLoad(test_split,
                                               target_label='cluster',
                                               exclude_missing_file=False,
                                               workdir=workdir)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    else:
        raise Exception(f'Invalid loader_type:{loader_type}')

    return train_loader, val_loader, test_loader
