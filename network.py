import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertModel
import torchaudio
import numpy as np


class AudioNet(nn.Module):
    def __init__(self):
        super(AudioNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(32, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
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
        input_shape = (64, 1292)

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
