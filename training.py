import json
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.tensorboard import SummaryWriter

from data_loader import get_data_loader
from network import MultiNet


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

    train_start_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else ''
    nets = sys.argv[2] if len(sys.argv) > 2 else 'al'
    # 'a' stands for audio and 'l' stand for lyrics

    writer = SummaryWriter(log_dir=os.path.join(dataset_dir, f'runs/{train_start_time}'))

    # Hyperparameters
    hparams = {
        'validation_split': 0.2,
        'test_split': 0,
        'batch_size': 256,
        'learning_rate': 3e-4,
        'weight_decay': 0.003,
        'num_epochs': 1000,
    }

    train_loader, val_loader, _, classes_name, vocab, embeddings = get_data_loader(
        validation_split=hparams['validation_split'],
        test_split=hparams['test_split'],
        batch_size=hparams['batch_size'],
        dataset_dir=dataset_dir)

    # Build model
    model = MultiNet(nets=nets, vocab_size=len(vocab), embeddings=embeddings)
    # model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)

    # Load pretrained model
    # model.load_state_dict(torch.load(os.path.join(dataset_dir, 'saved_models/model-1654583400.2823615(best).pt'),
    #                                  map_location=torch.device('cpu')))

    # Init optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hparams['learning_rate'],
                                 weight_decay=hparams['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()

    model = model.to(device)

    # Save model to tensorboard
    model_summary = str(model).replace('\n', '<br/>').replace(' ', '&nbsp;')
    writer.add_text("model", model_summary, 0)

    # Save hyperparameter to tensorboard
    writer.add_text("hyper-parameters",
                    json.dumps({**hparams, "optim": str(optimizer), "loss": str(criterion)}, indent=4,
                               sort_keys=True).replace('\n', '<br/>').replace(' ', '&nbsp;'), 0)

    loss_log = []
    acc_log = []
    val_acc_log = []
    val_loss_log = []
    max_acc_so_far = -1

    # make directory to store models
    os.makedirs(os.path.join(dataset_dir, f'saved_models/{train_start_time}'), exist_ok=True)

    # Main training loop
    for i in range(hparams['num_epochs']):

        # Run an epoch of training
        train_running_loss = 0
        train_running_acc = 0
        model.train()
        for j, (spec, lyric, label) in enumerate(train_loader, 0):
            spec, lyric, label = spec.to(device), lyric.to(device), label.to(device)

            out = model(spec, lyric)
            # out = model(input_ids=lyric['input_ids'].squeeze(1),
            #             attention_mask=lyric['attention_mask'],
            #             return_dict=False)[0]

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
        confusion_matrix = np.zeros((4, 4))
        model.eval()
        for j, input in enumerate(val_loader, 0):
            spec, lyric, label = input
            spec, lyric, label = spec.to(device), lyric.to(device), label.to(device)

            out = model(spec, lyric)
            # out = model(input_ids=lyric['input_ids'].squeeze(1),
            #             attention_mask=lyric['attention_mask'],
            #             return_dict=False)[0]

            loss = criterion(out, label)
            _, predicted = torch.max(out.data, 1)
            correct = (predicted == label).sum()

            val_acc += correct.item()
            val_loss += loss.item()

            for t, p in zip(label, predicted):
                confusion_matrix[t, p] += 1

        val_acc /= len(val_loader.dataset)
        val_loss /= j

        classes_name_only = list(zip(*sorted(classes_name.items(), key=lambda x: x[0])))[1]
        df_cm = pd.DataFrame(confusion_matrix, index=classes_name_only, columns=classes_name_only).astype(int)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=90, ha='right', fontsize=15)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='center', fontsize=15)
        plt.ylabel('True class')
        plt.xlabel('Predicted class')

        writer.add_figure("Confusion matrix", plt.gcf(), global_step=i)

        # Save models
        if val_acc > max_acc_so_far:
            max_acc_so_far = val_acc
            torch.save(model.state_dict(),
                       os.path.join(dataset_dir, f'saved_models/{train_start_time}/model-{time.time()}.pt'))

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

        # for name, weight in model.named_parameters():
        #     writer.add_histogram(name, weight, i)
        #     writer.add_histogram(f'{name}.grad', weight.grad, i)

    writer.close()


if __name__ == '__main__':
    main()
