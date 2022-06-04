# -*- coding: utf-8 -*-
"""text_word_embedded_multiple_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17umIQGXCYrHxJnaeXSvitP0grqiMtwZq
"""

# Commented out IPython magic to ensure Python compatibility.

import numpy as np
import pandas as pd
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import itertools
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import math
import collections
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from scipy import interpolate
from scipy.spatial import ConvexHull
from scipy import stats
import matplotlib.pyplot as plt
# %matplotlib inline
import json
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report,precision_recall_fscore_support

### Helper functions
import pickle

def convert_to_pickle(item, directory):
    pickle.dump(item, open(directory,"wb"))


def load_from_pickle(directory):
    return pickle.load(open(directory,"rb"))

SEED = 1234
def set_seeds(seed=1234):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # multi-GPU
# Set seeds for reproducibility
set_seeds(seed=SEED)

# Set device
cuda = True
device = torch.device("cuda" if (
    torch.cuda.is_available() and cuda) else "cpu")
torch.set_default_tensor_type("torch.FloatTensor")
if device.type == "cuda":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
# print (device)

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/gdrive')
# %cd /gdrive/MyDrive/DL_Project/dataset

# Load data
csv1 = "dataset.csv"
csv2 = "lyrics.csv"
df1 = pd.read_csv(csv1, header=0) # load
df2 = pd.read_csv(csv2, header=0) # load
#df = df.sample(frac=1).reset_index(drop=True) # shuffle
#df1.head(5)
#df2.head(5)

# frames = [df1, df2]
# result = pd.concat(frames)
# display(result)

frames = pd.merge(df1, df2, on=['dzr_sng_id'])
#display(frames)

nltk.download("stopwords")
STOPWORDS = stopwords.words("english")
print (STOPWORDS[:5])
porter = PorterStemmer()

def preprocess(text, stopwords=STOPWORDS):
    """Conditional preprocessing on our text unique to our task."""
    # Lower
    text = text.lower()

    # Remove stopwords
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub("", text)

    # Remove words in parenthesis
    text = re.sub(r"\([^)]*\)", "", text)

    # Spacing and filters
    text = re.sub(r"([-;;.,!?<=>])", r" \1 ", text)
    text = re.sub("[^A-Za-z0-9]+", " ", text) # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()

    return text

# Apply to dataframe
preprocessed_df = frames.copy()
preprocessed_df.lyrics = preprocessed_df.lyrics.apply(preprocess)
print (f"{frames.lyrics.values[0]}\n\n{preprocessed_df.lyrics.values[0]}")

TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

def train_val_test_split(X, y, train_size):
    """Split dataset into data splits."""
    X_train, X_, y_train, y_ = train_test_split(X, y, train_size=TRAIN_SIZE, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5, stratify=y_)
    return X_train, X_val, X_test, y_train, y_val, y_test

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

preprocessed_df['cluster'] = create_cluster(preprocessed_df, num_clusters=4)
#num classes need to be defined earlier
display(preprocessed_df)

# retain only text that contain less that 70 tokens to avoid too much padding
preprocessed_df["token_size"] = preprocessed_df["lyrics"].apply(lambda x: len(x.split(' ')))
preprocessed_data = preprocessed_df.loc[preprocessed_df['token_size'] < 70].copy()

# Data
X = preprocessed_data["lyrics"].values
y = preprocessed_data["cluster"].values
#print(len(X[17].split(" ")))

preprocessed_df.cluster.value_counts().plot.bar()

print(preprocessed_data)

# This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa 
# (e.g., 5 -> "dad") for the dataset
class ConstructVocab():
    def __init__(self, sentences):
        self.sentences = sentences
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()
        
    def create_index(self):
        for s in self.sentences:
            # update with individual tokens
            self.vocab.update(s.split(' '))
            
        # sort the vocab
        self.vocab = sorted(self.vocab)

        # add a padding token with index 0
        self.word2idx['<pad>'] = 0
        
        # word to index mapping
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1 # +1 because of pad token
        
        # index to word mapping
        for word, index in self.word2idx.items():
            self.idx2word[index] = word

# construct vocab and indexing
inputs = ConstructVocab(preprocessed_data["lyrics"].values.tolist())

# examples of what is in the vocab
#inputs.vocab[0:10] #a little issue

# vectorize to tensor
input_tensor = [[inputs.word2idx[s] for s in es.split(' ')]  for es in preprocessed_data["lyrics"].values.tolist()]
# examples of what is in the input tensors
#input_tensor[0:2]
print(len(input_tensor[0]))

#padding to input layer
def pad_sequences(sequences, max_seq_len=0):
    """Pad sequences to max length in sequence."""
    max_seq_len = max(max_seq_len, max(len(sequence) for sequence in sequences))
    padded_sequences = np.zeros((len(sequences), max_seq_len))
    for i, sequence in enumerate(sequences):
        padded_sequences[i][:len(sequence)] = sequence
    return padded_sequences

padded = pad_sequences(input_tensor[0:3])
print (padded.shape)
#print (padded)

#dataset
FILTER_SIZES = list(range(1, 4)) # uni, bi and tri grams

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, max_filter_size):
        self.X = X
        self.y = y
        self.max_filter_size = max_filter_size

    def __len__(self):
        return len(self.y)

    def __str__(self):
        return f"<Dataset(N={len(self)})>"

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return [X, y]

    def collate_fn(self, batch):
        """Processing on a batch."""
        # Get inputs
        batch = np.array(batch)
        X = batch[:, 0]
        y = batch[:, 1]

        # Pad sequences
        X = pad_sequences(X)

        # Cast
        X = torch.LongTensor(X.astype(np.int32))
        y = torch.LongTensor(y.astype(np.int32))

        return X, y

    def create_dataloader(self, batch_size, shuffle=False, drop_last=False):
        return torch.utils.data.DataLoader(
            dataset=self, batch_size=batch_size, collate_fn=self.collate_fn,
            shuffle=shuffle, drop_last=drop_last, pin_memory=True)

# Create data splits
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X=input_tensor, y=y, train_size=TRAIN_SIZE)
print (f"X_train: {len(X_train)}, y_train: {len(y_train)}")
print (f"X_val: {len(X_val)}, y_val: {len(y_val)}")
print (f"X_test: {len(X_test)}, y_test: {len(y_test)}")
print (f"Sample point: {X_train[0]} → {y_train[0]}")

# Create datasets
max_filter_size = max(FILTER_SIZES)
train_dataset = Dataset(X=X_train, y=y_train, max_filter_size=max_filter_size)
val_dataset = Dataset(X=X_val, y=y_val, max_filter_size=max_filter_size)
test_dataset = Dataset(X=X_test, y=y_test, max_filter_size=max_filter_size)
print ("Datasets:\n"
    f"  Train dataset:{train_dataset.__str__()}\n"
    f"  Val dataset: {val_dataset.__str__()}\n"
    f"  Test dataset: {test_dataset.__str__()}\n"
    "Sample point:\n"
    f"  X: {train_dataset[0][0]}\n"
    f"  y: {train_dataset[0][1]}")

# Create dataloaders
batch_size = 64
train_dataloader = train_dataset.create_dataloader(batch_size=batch_size, drop_last=True)
val_dataloader = val_dataset.create_dataloader(batch_size=batch_size, drop_last=True)
test_dataloader = test_dataset.create_dataloader(batch_size=batch_size, drop_last=True)
batch_X, batch_y = next(iter(train_dataloader))
print ("Sample batch:\n"
    f"  X: {list(batch_X.size())}\n"
    f"  y: {list(batch_y.size())}\n"
    "Sample point:\n"
    f"  X: {batch_X[0]}\n"
    f"  y: {batch_y[0]}")

EMBEDDING_DIM = 100
HIDDEN_DIM = 100
DROPOUT_P = 0.1

class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab_size, num_filters,
                 filter_sizes, hidden_dim, dropout_p, num_classes,
                 pretrained_embeddings=None, freeze_embeddings=False,
                 padding_idx=0):
        super(CNN, self).__init__()

        # Filter sizes
        self.filter_sizes = filter_sizes

        # Initialize embeddings
        if pretrained_embeddings is None:
            self.embeddings = nn.Embedding(
                embedding_dim=embedding_dim, num_embeddings=vocab_size,
                padding_idx=padding_idx)
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.embeddings = nn.Embedding(
                embedding_dim=embedding_dim, num_embeddings=vocab_size,
                padding_idx=padding_idx, _weight=pretrained_embeddings)

        # Freeze embeddings or not
        if freeze_embeddings:
            self.embeddings.weight.requires_grad = False

        # Conv weights
        self.conv = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim,
                       out_channels=num_filters,
                       kernel_size=f) for f in filter_sizes])

        # FC weights
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(num_filters*len(filter_sizes), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs, channel_first=False):

        # Embed
        x_in, = inputs
        x_in = self.embeddings(x_in)

        # Rearrange input so num_channels is in dim 1 (N, C, L)
        if not channel_first:
            x_in = x_in.transpose(1, 2)

        # Conv outputs
        z = []
        max_seq_len = x_in.shape[2]
        for i, f in enumerate(self.filter_sizes):
            # `SAME` padding
            padding_left = int((self.conv[i].stride[0]*(max_seq_len-1) - max_seq_len + self.filter_sizes[i])/2)
            padding_right = int(math.ceil((self.conv[i].stride[0]*(max_seq_len-1) - max_seq_len + self.filter_sizes[i])/2))

            # Conv + pool
            _z = self.conv[i](F.pad(x_in, (padding_left, padding_right)))
            _z = F.max_pool1d(_z, _z.size(2)).squeeze(2)
            z.append(_z)

        # Concat conv outputs
        z = torch.cat(z, 1)

        # FC layers
        z = self.fc1(z)
        z = self.dropout(z)
        z = self.fc2(z)
        return z

class GRU_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units, batch_sz, output_size, 
             pretrained_embeddings, freeze_embeddings, padding_idx=0):
        super(GRU_Model, self).__init__()
        self.batch_sz = batch_sz
        self.hidden_units = hidden_units
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.output_size = output_size

        #initialise embeddings
        pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
        self.embeddings = nn.Embedding(
            embedding_dim=embedding_dim, num_embeddings=vocab_size,
            padding_idx=padding_idx, _weight=pretrained_embeddings)
        
        # Freeze embeddings or not
        if freeze_embeddings:
            self.embeddings.weight.requires_grad = False

        # layers
        # self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(p=0.25)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_units, num_layers=2,bidirectional=True,batch_first=True)
        self.fc = nn.Linear(2*self.hidden_units, self.output_size)

    def initialize_hidden_state(self, device):
        return torch.zeros((4, self.batch_sz, self.hidden_units)).to(device)

    def forward(self, inputs):
        x, = inputs
        print(x.shape)
        x = self.embeddings(x)
        print(x.shape)
        self.hidden = self.initialize_hidden_state(device)
        output, self.hidden = self.gru(x, self.hidden) # max_len X batch_size X hidden_units

        out = self.dropout(output)
        out = output[:, -1, :] 
        out = self.fc(out)
        print(out.shape)
        return out

class LSTM_model(nn.Module):
    def __init__(self, batch_sz, vocab_size, embedding_dim, output_size, pretrained_embeddings,
                 freeze_embeddings, padding_idx=0, hidden_dim = 64) :
        # super().__init__()
        super(LSTM_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_sz = batch_sz
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.output_size = output_size

        #initialise embeddings
        pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
        self.embeddings = nn.Embedding(
            embedding_dim=embedding_dim, num_embeddings=vocab_size,
            padding_idx=padding_idx, _weight=pretrained_embeddings)
        
        # Freeze embeddings or not
        if freeze_embeddings:
            self.embeddings.weight.requires_grad = False

        # layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(p=0.25)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True) #x -- batch_size first
        self.fc = nn.Linear(self.hidden_dim, self.output_size) #output -- 2 (even/odd)
        # self.activation_func = nn.Sigmoid()

    def initialize_hidden_state(self, device):
        return torch.zeros((4, self.batch_sz, self.hidden_units)).to(device)

    def forward(self, inputs):
        # Embed
        x_in, = inputs
        x_in = self.embeddings(x_in)
        lstm_output, (h, c) = self.lstm(x_in, None)
        out = lstm_output[:, -1, :] 
        out = self.fc(out)
        return out

#Using GloVe
def load_glove_embeddings(embeddings_file):
    """Load embeddings from a file."""
    embeddings = {}
    with open(embeddings_file, "r") as fp:
        for index, line in enumerate(fp):
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings[word] = embedding
    return embeddings

def make_embeddings_matrix(embeddings, word_index, embedding_dim):
    """Create embeddings matrix to use in Embedding layer."""
    embedding_matrix = np.zeros((len(word_index), embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# Create embeddings
embeddings_file = 'glove.6B.100d.txt'.format(EMBEDDING_DIM)
glove_embeddings = load_glove_embeddings(embeddings_file=embeddings_file)
embedding_matrix = make_embeddings_matrix(embeddings=glove_embeddings,
      word_index=inputs.word2idx, embedding_dim=EMBEDDING_DIM) #tokenizer.token_to_index
print (f"<Embeddings(words={embedding_matrix.shape[0]}, dim={embedding_matrix.shape[1]})>")

#Experiments
NUM_FILTERS = 50
LEARNING_RATE = 1e-3
PATIENCE = 5
NUM_EPOCHS = 10

class Trainer(object):
    def __init__(self, model, device, loss_fn=None, optimizer=None, scheduler=None):

        # Set params
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_step(self, dataloader):
        """Train step."""
        # Set model to train mode
        self.model.train()
        loss = 0.0

        # Iterate over train batches
        for i, batch in enumerate(dataloader):
            # if len(batch) < 64:
            #   break
            # Step
            batch = [item.to(self.device) for item in batch]  # Set device
            inputs, targets = batch[:-1], batch[-1]
            print(targets.shape)

            self.optimizer.zero_grad()  # Reset gradients
            z = self.model(inputs)  # Forward pass
            print(z.shape)
            J = self.loss_fn(z, targets)  # Define loss
            J.backward()  # Backward pass
            self.optimizer.step()  # Update weights
            #print("done")

            # Cumulative Metrics
            loss += (J.detach().item() - loss) / (i + 1)

        return loss

    def eval_step(self, dataloader):
        """Validation or test step."""
        # Set model to eval mode
        self.model.eval()
        loss = 0.0
        y_trues, y_probs = [], []

        # Iterate over val batches
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                # if len(batch) < 64:
                #   return loss,[],[]
                # Step
                batch = [item.to(self.device) for item in batch]  # Set device
                inputs, y_true = batch[:-1], batch[-1]
                z = self.model(inputs)  # Forward pass
                J = self.loss_fn(z, y_true).item()

                # Cumulative Metrics
                loss += (J - loss) / (i + 1)

                # Store outputs
                y_prob = F.softmax(z).cpu().numpy()
                y_probs.extend(y_prob)
                y_trues.extend(y_true.cpu().numpy())

        return loss, np.vstack(y_trues), np.vstack(y_probs)

    def predict_step(self, dataloader):
        """Prediction step."""
        # Set model to eval mode
        self.model.eval()
        y_probs = []

        # Iterate over val batches
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):

                # Forward pass w/ inputs
                inputs, targets = batch[:-1], batch[-1]
                z = self.model(inputs)

                # Store outputs
                y_prob = F.softmax(z).cpu().numpy()
                y_probs.extend(y_prob)

        return np.vstack(y_probs)

    def train(self, num_epochs, patience, train_dataloader, val_dataloader):
        best_val_loss = np.inf
        for epoch in range(num_epochs):
            # Steps
            train_loss = self.train_step(dataloader=train_dataloader)
            val_loss, _, _ = self.eval_step(dataloader=val_dataloader)
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model
                _patience = patience  # reset _patience
            else:
                _patience -= 1
            if not _patience:  # 0
                print("Stopping early!")
                break

            # Logging
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.5f}, "
                f"val_loss: {val_loss:.5f}, "
                f"lr: {self.optimizer.param_groups[0]['lr']:.2E}, "
                f"_patience: {_patience}"
            )
        return best_model

def get_metrics(y_true, y_pred, classes):
    """Per-class performance metrics."""
    # Performance
    performance = {"overall": {}, "class": {}}

    # Overall performance
    metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    performance["overall"]["precision"] = metrics[0]
    performance["overall"]["recall"] = metrics[1]
    performance["overall"]["f1"] = metrics[2]
    performance["overall"]["num_samples"] = np.float64(len(y_true))

    # Per-class performance
    metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i in range(len(classes)):
        performance["class"][classes[i]] = {
            "precision": metrics[0][i],
            "recall": metrics[1][i],
            "f1": metrics[2][i],
            "num_samples": np.float64(metrics[3][i]),
        }

    return performance

PRETRAINED_EMBEDDINGS = embedding_matrix
FREEZE_EMBEDDINGS = True

embedding_matrix.shape

len(inputs.vocab)

# Initialize model
1
model = CNN(
    embedding_dim=EMBEDDING_DIM, vocab_size=len(inputs.word2idx),
    num_filters=NUM_FILTERS, filter_sizes=FILTER_SIZES,
    hidden_dim=HIDDEN_DIM, dropout_p=DROPOUT_P, num_classes=4,
    pretrained_embeddings=PRETRAINED_EMBEDDINGS, freeze_embeddings=FREEZE_EMBEDDINGS)

model = model.to(device) # set device
print (model.named_parameters)

# 2 #output_size = num_classes
model = GRU_Model(vocab_size=len(inputs.word2idx), embedding_dim=EMBEDDING_DIM, hidden_units=100, batch_sz=batch_size, output_size=4, pretrained_embeddings=PRETRAINED_EMBEDDINGS, freeze_embeddings=FREEZE_EMBEDDINGS, padding_idx=0)

model = model.to(device) # set device
print (model.named_parameters)

# 3 
model = LSTM_model(batch_sz=batch_size, vocab_size=len(inputs.word2idx), embedding_dim=EMBEDDING_DIM,
    output_size=4, pretrained_embeddings=PRETRAINED_EMBEDDINGS, freeze_embeddings=FREEZE_EMBEDDINGS, padding_idx=0, hidden_dim = 64)

model = model.to(device) # set device
print (model.named_parameters)

# Define Loss
#class_weights_tensor = torch.Tensor(list(class_weights.values())).to(device)
#loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
loss_fn = nn.CrossEntropyLoss()

# Define optimizer & scheduler
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=3)

# Trainer module
trainer = Trainer(
    model=model, device=device, loss_fn=loss_fn,
    optimizer=optimizer, scheduler=scheduler)

# Train
best_model = trainer.train(NUM_EPOCHS, PATIENCE, train_dataloader, val_dataloader)

# Get predictions
test_loss, y_true, y_prob = trainer.eval_step(dataloader=test_dataloader)
print(y_prob)
y_pred = np.argmax(y_prob, axis=1)

def evaluate(targets, predictions):
    performance = {
        'acc': accuracy_score(targets, predictions),
        'f1': f1_score(targets, predictions, average='macro'),
        'precision': precision_score(targets, predictions, average='macro'),
        'recall': recall_score(targets, predictions, average='macro')
    }
    return performance

def plot_confusion_matrix(targets,
                          predictions,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # plt.figure(figsize=(8,8))
    cm = confusion_matrix(targets, predictions)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,
                 i,
                 cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def calc_test_result(predicted_label, true_label):

		print("Confusion Matrix :")
		print(confusion_matrix(true_label, predicted_label))
		print("Classification Report :")
		print(classification_report(true_label, predicted_label, digits=4))
		print('Weighted FScore: \n ', precision_recall_fscore_support(true_label, predicted_label, average='weighted'))

calc_test_result(y_pred, y_true)