import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel


class LyricsDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.lyrics = df['lyrics'].tolist()
        self.lyrics = [tokenizer(lyric,
                                 padding='max_length',
                                 max_length=64,
                                 truncation=True,
                                 return_tensors="pt") for lyric in self.lyrics]

        self.labels = df['label'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        lyric = self.lyrics[idx]
        label = torch.tensor(self.labels[idx])
        return lyric, label


class LyricsNet(nn.Module):
    def __init__(self, dropout=0.5):
        super(LyricsNet, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, out_features=4)

    def forward(self, input_id, mask):
        _, out = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        out = self.dropout(out)
        out = self.linear(out)
        out = F.relu(out)
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

    # storage_dir = '/nfs/stak/users/panditb/storage/workspace/SongMultimodelAnalysis/dataset'
    storage_dir = '../dataset'
    df = pd.read_csv(os.path.join(storage_dir, 'dataset.csv'))
    df = df.merge(pd.read_csv(os.path.join(storage_dir, 'lyrics.csv')), on='dzr_sng_id')

    df = df[:2000]

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    size = len(df)
    df_train = df[:int(size * 0.8)]
    df_val = df[int(size * 0.8):]

    train_data = LyricsDataset(df_train, tokenizer)
    val_data = LyricsDataset(df_val, tokenizer)

    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=2)

    # Build model
    model = LyricsNet()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-6)
    criterion = torch.nn.CrossEntropyLoss()

    model = model.to(device)

    loss_log = []
    acc_log = []
    val_acc_log = []
    val_loss_log = []
    max_acc_so_far = -1

    for i in range(100):

        # Run an epoch of training
        train_running_loss = 0
        train_running_acc = 0
        model.train()
        for j, input in enumerate(train_loader, 0):
            x = input[0]['input_ids'].squeeze(1).to(device)
            mask = input[0]['attention_mask'].to(device)
            y = input[1].to(device)
            out = model.forward(x, mask)
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
            x = input[0]['input_ids'].squeeze(1).to(device)
            mask = input[0]['attention_mask'].to(device)
            y = input[1].to(device)

            out = model(x, mask)

            loss = criterion(out, y)
            _, predicted = torch.max(out.data, 1)
            correct = (predicted == y).sum()

            val_acc += correct.item()
            val_loss += loss.item()

        val_acc /= len(val_loader.dataset)
        val_loss /= j

        val_acc_log.append(val_acc)
        val_loss_log.append(val_loss)

        print(
            str(time.time()) + " [Epoch {:3}]   Loss:  {:8.4}  Val Loss:  {:8.4}  Train Acc:  {:8.4}%      Val Acc:  {:8.4}%".format(
                i,
                train_running_loss,
                val_loss,
                train_running_acc * 100,
                val_acc * 100))


if __name__ == '__main__':
    main()
