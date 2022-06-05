import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from data_loader import get_data_loader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 1-channel
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )

        x = self.feature_extractor(torch.rand(64, 1292))
        in_features = np.prod(x.shape)
        print(f'Calculated in_features={in_features}')

        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=4),
            nn.Sigmoid()
        )

    def forward(self, spec, lyric):
        out = self.feature_extractor(spec)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


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

    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else ''

    writer = SummaryWriter()

    # Hyperparameters
    validation_split = 0.1
    batch_size = 64
    learning_rate = 1e-4
    weight_decay = 0.03

    # Warning: Re-clustering might change the order of classes
    classes_name = {0: 'HV-LA',
                    1: 'HV-HA',
                    2: 'LV-LA',
                    3: 'LV-HA'}

    train_loader, val_loader, _ = get_data_loader(validation_split=validation_split,
                                                  test_split=0,
                                                  batch_size=batch_size,
                                                  classes_name=classes_name,
                                                  dataset_dir=dataset_dir)

    # Build model
    model = Net()

    # model = resnet34(pretrained=True)
    # model.fc = nn.Linear(512, 50)
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Load saved models
    # model.load_state_dict(torch.load(
    #     os.path.join('saved_models', 'model-1653693254.657536.pt'),
    #     map_location=device))

    # Init optimizer and loss function
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
    os.makedirs(os.path.join(dataset_dir, 'saved_models'), exist_ok=True)

    # Main training loop
    for i in range(100):

        # Run an epoch of training
        train_running_loss = 0
        train_running_acc = 0
        model.train()
        for j, input in enumerate(train_loader, 0):
            spec, lyric, label = input
            spec, lyric, label = spec.to(device), lyric.to(device), label.to(device)

            out = model(spec, lyric)

            loss = criterion(out, label)

            model.zero_grad()
            loss.backward()

            optimizer.step()

            _, predicted = torch.max(out.data, 1)
            correct = (predicted == label).sum()

            train_running_loss += loss.item()
            train_running_acc += correct.item()

            loss_log.append(loss.item())
            acc_log.append(correct.item() / len(label))

        train_running_loss /= j
        train_running_acc /= len(train_loader.dataset)

        # Evaluate on validation
        val_acc = 0
        val_loss = 0
        model.eval()
        for j, input in enumerate(val_loader, 0):
            spec, lyric, label = input
            spec, lyric, label = spec.to(device), lyric.to(device), label.to(device)

            out = model(spec, lyric)

            loss = criterion(out, label)
            _, predicted = torch.max(out.data, 1)
            correct = (predicted == label).sum()

            val_acc += correct.item()
            val_loss += loss.item()

        val_acc /= len(val_loader.dataset)
        val_loss /= j

        # Save models
        if val_acc > max_acc_so_far:
            max_acc_so_far = val_acc
            torch.save(model.state_dict(), os.path.join(dataset_dir, f'saved_models/model-{time.time()}.pt'))

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


if __name__ == '__main__':
    main()
