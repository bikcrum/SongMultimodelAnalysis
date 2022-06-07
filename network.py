import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertModel
import torchaudio
import numpy as np

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
class AudioNet(nn.Module):
    def __init__(self):
        super(AudioNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4,4)),

            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4,4)),

            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4,4)),
        )

    def compute_flat_feature(self, shape):
        out = self.layers(torch.rand(shape))
        num_features = np.prod(out.shape)
        return num_features

    def forward(self, spec):
        out = self.layers(spec)
        out = out.view(out.size(0), -1)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_length=5000):
        super(PositionalEncoding, self).__init__()

        pos = torch.arange(0, max_length).unsqueeze(1)
        pos_encoding = torch.zeros((max_length, d_model))

        sin_den = 10000 ** (torch.arange(0, d_model, 2) / d_model)
        cos_den = 10000 ** (torch.arange(1, d_model, 2) / d_model)

        pos_encoding[:, 0::2] = torch.sin(pos / sin_den)
        pos_encoding[:, 1::2] = torch.cos(pos / cos_den)

        pos_encoding = pos_encoding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(InputEmbedding, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens):
        inp_emb = self.embedding(tokens.long()) * np.sqrt(self.d_model)
        return inp_emb


class LyricNet(nn.Module):
    def __init__(self, vocab_size):
        super(LyricNet, self).__init__()
        self.embedding = InputEmbedding(vocab_size=vocab_size, d_model=512)

        self.positional_encoding = PositionalEncoding(d_model=512, dropout=0.1)

        encoder_layers = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=128, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)

    def compute_flat_feature(self, shape):
        # d_model
        return 512

    def forward(self, lyric):
        embedding = self.positional_encoding(self.embedding(lyric))

        output = self.transformer_encoder(embedding)

        # Mean over sequence length
        return output.mean(axis=1)


class MultiNet(nn.Module):
    def __init__(self, nets='al', vocab_size=None):
        super(MultiNet, self).__init__()

        num_features = 0
        input_shape = (1, 64, 1292)

        self.nets = nets

        if 'a' in nets:
            self.audio_net = AudioNet()
            num_features += self.audio_net.compute_flat_feature(input_shape)

        if 'l' in nets:
            self.lyric_net = LyricNet(vocab_size=vocab_size)
            num_features += self.lyric_net.compute_flat_feature(input_shape)

        print(f'Received nets:{nets}')
        print(f'Total flat features:{num_features}')

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=128),
            nn.Linear(in_features=128, out_features=4)
        )

    def forward(self, spec, lyric):
        if 'a' in self.nets and 'l' in self.nets:
            feature1 = self.audio_net(spec)
            feature2 = self.lyric_net(lyric)
            feature = torch.cat((feature1, feature2), dim=1)
        elif 'a' in self.nets:
            feature = self.audio_net(spec)
        elif 'l' in self.nets:
            feature = self.lyric_net(lyric)
        else:
            raise Exception('At least one model must be initialized')

        out = self.classifier(feature)

        return out
