import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from data_loader import get_data_loader


class Net(nn.Module):
    def __init__(self, n_mels, num_classes):
        super(Net, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(n_mels, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(4))

        self.conv1 = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(4))

        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(8, 4, kernel_size=3),
        #     nn.ReLU(),
        #     nn.MaxPool1d(4),
        # )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=160, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=num_classes))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            # For apple silicon
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        except:
            device = torch.device("cpu")

    print(f"Using device: {device}")

    workdir = sys.argv[1] if len(sys.argv) > 1 else ''

    writer = SummaryWriter()

    # Hyperparameters
    num_classes = 4
    validation_split = 0.2
    batch_size = 256
    n_mels = 64
    learning_rate = 1e-4
    weight_decay = 0.003

    train_loader, val_loader, _ = get_data_loader(validation_split=validation_split,
                                                  num_classes=num_classes,
                                                  batch_size=batch_size,
                                                  loader_type='lazy_load',
                                                  workdir=workdir)

    # mean = train_data.xx.mean(axis=1).mean(axis=0)
    # std = train_data.xx.std(axis=1).std(axis=0)

    # mean = train_data.xx.mean(axis=0)
    # std = train_data.xx.std(axis=0)

    # train_data.xx = (train_data.xx - mean) / std
    # val_data.xx = (val_data.xx - mean) / std
    # test_data.xx = (test_data.xx - mean) / std

    # Build model
    model = Net(n_mels=n_mels, num_classes=num_classes)
    # model.load_state_dict(torch.load(
    #     os.path.join('saved_models', 'model-1653693254.657536.pt'),
    #     map_location=device))

    # Main training loop
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    model = model.to(device)

    loss_log = []
    acc_log = []
    val_acc_log = []
    val_loss_log = []
    max_acc_so_far = -1

    # make directory to store models
    os.makedirs(os.path.join(workdir, 'saved_models'), exist_ok=True)

    for i in range(200):

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
        train_running_acc /= len(train_loader.dataset)

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

        val_acc /= len(val_loader.dataset)
        val_loss /= j

        # Save models
        if val_acc > max_acc_so_far:
            max_acc_so_far = val_acc
            torch.save(model.state_dict(), os.path.join(workdir, f'saved_models/model-{time.time()}.pt'))

        val_acc_log.append(val_acc)
        val_loss_log.append(val_loss)

        print(
            str(time.time()) + " [Epoch {:3}]   Loss:  {:8.4}  Val Loss:  {:8.4}  Train Acc:  {:8.4}%      Val Acc:  {:8.4}%".format(
                i,
                train_running_loss,
                val_loss,
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
    ax1.plot([np.ceil((i + 1) * len(train_loader.dataset) / batch_size) for i in range(len(val_loss_log))],
             val_loss_log, c="red",
             label="Val. Loss")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.01, 3)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.plot(range(len(acc_log)), acc_log, c=color, label="Train Acc.", alpha=0.25)
    ax2.plot([np.ceil((i + 1) * len(train_loader.dataset) / batch_size) for i in range(len(val_acc_log))], val_acc_log,
             c="blue",
             label="Val. Acc.")
    ax2.set_ylabel(" Accuracy", c=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.01, 1.01)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend(loc="center")
    ax2.legend(loc="center right")
    plt.show()


if __name__ == '__main__':
    main()
