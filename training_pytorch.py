import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import interpolate
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = "cpu"

print(f"Using device: {device}")


def create_cluster(num_clusters=None):
    x = dataset.loc[:, ['valence', 'arousal']]

    if num_clusters:
        kmeans = KMeans(num_clusters)
        kmeans.fit(dataset[['valence', 'arousal']])
        clusters = kmeans.fit_predict(x)
        dataset['cluster'] = clusters

    fig, ax = plt.subplots(1, figsize=(8, 8))

    plt.scatter(dataset.valence, dataset.arousal, c=dataset.cluster)

    for i in dataset.cluster.unique():
        points = dataset[dataset.cluster == i][['valence', 'arousal']].values
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


class DeezerMusicDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.xx = []
        for song_id in dataset.dzr_sng_id:
            with open(f'dataset/previews/melspectrogram/{song_id}.mel', 'rb') as r:
                self.xx.append(pickle.load(r))
            # self.xx.append(np.load(f'dataset/previews/melspectrogram/{song_id}.mel'))

        self.xx = np.array(self.xx)
        self.yy = dataset[['cluster']].squeeze().values

        self.transform = transform

    def __len__(self):
        return self.yy.shape[0]

    def __getitem__(self, idx):
        x = Tensor(self.xx[idx])

        if self.transform is not None:
            x = self.transform(x)

        y = self.yy[idx]

        return x, y


class Net(nn.Module):
    def __init__(self, n_mels, num_clusters):
        super(Net, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(n_mels, 32, kernel_size=3),
            nn.MaxPool1d(4))

        self.conv1 = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3),
            nn.MaxPool1d(4))

        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=3),
            nn.MaxPool1d(4))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=160, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=num_clusters))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    num_clusters = 4

    writer = SummaryWriter()

    dataset = pd.read_csv('dataset/dataset.csv')
    # dataset['valence'] = np.random.randn(len(dataset))
    # dataset['arousal'] = np.random.randn(len(dataset))
    create_cluster(num_clusters=4)

    train_split, val_split, test_split = split_data(dataset, splits=[0.2, 0.1])
    print(len(train_split), len(val_split), len(test_split))
    train_data = DeezerMusicDataset(train_split)
    val_data = DeezerMusicDataset(val_split)
    test_data = DeezerMusicDataset(test_split)

    batch_size = 256
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Build model
    model = Net(n_mels=64, num_clusters=num_clusters)
    # model.load_state_dict()

    # Main training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.003)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    model.to(device)

    loss_log = []
    acc_log = []
    val_acc_log = []
    val_loss_log = []
    max_acc_so_far = -1

    # make directory to store models
    os.makedirs('saved_models', exist_ok=True)

    for i in range(100):

        # Run an epoch of training
        train_running_loss = 0
        train_running_acc = 0
        model.train()
        for j, input in enumerate(train_loader, 0):
            x = input[0].to(device)
            y = input[1].type(torch.LongTensor).to(device)
            out = model(x)
            loss = criterion(out, y)

            model.zero_grad()
            loss.backward()

            optimizer.step()

            _, predicted = torch.max(out.data, 1)
            correct = (predicted == y).sum()

            train_running_loss += loss.item()
            train_running_acc += correct.item()

            loss_log.append(loss.item())
            acc_log.append(correct.item() / len(y))

        train_running_loss /= j
        train_running_acc /= len(train_split)

        # Evaluate on validation
        val_acc = 0
        val_loss = 0
        model.eval()
        for j, input in enumerate(val_loader, 0):
            x = input[0].to(device)
            y = input[1].type(torch.LongTensor).to(device)

            out = model(x)

            loss = criterion(out, y)
            _, predicted = torch.max(out.data, 1)
            correct = (predicted == y).sum()

            val_acc += correct.item()
            val_loss += loss.item()

        val_acc /= len(val_split)
        val_loss /= j

        # Save models
        if val_acc > max_acc_so_far:
            max_acc_so_far = val_acc
            torch.save(model.state_dict(), f'saved_models/model-{time.time()}.pt')

        val_acc_log.append(val_acc)
        val_loss_log.append(val_loss)

        print(
            "[Epoch {:3}]   Loss:  {:8.4}  Val Loss:  {:8.4}  Train Acc:  {:8.4}%      Val Acc:  {:8.4}%".format(i,
                                                                                                                 val_loss,
                                                                                                                 train_running_loss,
                                                                                                                 train_running_acc * 100,
                                                                                                                 val_acc * 100))

        # Write results to tensorboard
        writer.add_scalar('Accuracy/train', train_running_acc * 100, i)
        writer.add_scalar('Accuracy/validation', val_acc * 100, i)
        writer.add_scalar('Loss/train', train_running_loss, i)
        writer.add_scalar('Loss/validation', val_loss, i)

        for name, weight in model.named_parameters():
            writer.add_histogram(name, weight, i)
            writer.add_histogram(f'{name}.grad', weight.grad, i)

    writer.close()

    # Plot training and validation curves
    fig, ax1 = plt.subplots(figsize=(16, 9))
    color = 'tab:red'
    ax1.plot(range(len(loss_log)), loss_log, c=color, alpha=0.25, label="Train Loss")
    ax1.plot([np.ceil((i + 1) * len(train_data) / batch_size) for i in range(len(val_loss_log))], val_loss_log, c="red",
             label="Val. Loss")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.01, 3)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.plot(range(len(acc_log)), acc_log, c=color, label="Train Acc.", alpha=0.25)
    ax2.plot([np.ceil((i + 1) * len(train_data) / batch_size) for i in range(len(val_acc_log))], val_acc_log, c="blue",
             label="Val. Acc.")
    ax2.set_ylabel(" Accuracy", c=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.01, 1.01)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend(loc="center")
    ax2.legend(loc="center right")
    plt.show()
