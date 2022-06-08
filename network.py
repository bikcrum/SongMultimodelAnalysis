import numpy as np
import torch
import torch.nn as nn


class AudioNet(nn.Module):
    def __init__(self):
        super(AudioNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(4, 4)),

            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(4, 4)),

            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(4, 4)),
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

    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, embeddings=None):
        super(InputEmbedding, self).__init__()

        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.d_model = d_model

    def forward(self, tokens):
        inp_emb = self.embedding(tokens.long()) * np.sqrt(self.d_model)
        return inp_emb


class LyricNet(nn.Module):
    def __init__(self, vocab_size, embeddings=None):
        super(LyricNet, self).__init__()

        if embeddings is not None:
            vocab_size, d_model = embeddings.size()
            self.d_model = d_model
        else:
            self.d_model = 512

        self.embedding = InputEmbedding(vocab_size=vocab_size, d_model=self.d_model, embeddings=embeddings)

        self.positional_encoding = PositionalEncoding(d_model=self.d_model, dropout=0.1, vocab_size=vocab_size)

        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=10, dim_feedforward=128, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)

    def compute_flat_feature(self, shape):
        # d_model
        return self.d_model

    def forward(self, lyric):
        embedding = self.positional_encoding(self.embedding(lyric))

        output = self.transformer_encoder(embedding)

        # Mean over sequence length
        return output.mean(axis=1)


class MultiNet(nn.Module):
    def __init__(self, nets='al', vocab_size=None, embeddings=None):
        super(MultiNet, self).__init__()

        num_features = 0
        input_shape = (1, 64, 1292)

        self.nets = nets

        if 'a' in nets:
            self.audio_net = AudioNet()
            num_features += self.audio_net.compute_flat_feature(input_shape)

        if 'l' in nets:
            self.lyric_net = LyricNet(vocab_size=vocab_size, embeddings=embeddings)
            num_features += self.lyric_net.compute_flat_feature(input_shape)

        print(f'Received nets:{nets}')
        print(f'Total flat features:{num_features}')

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=128),
            nn.Linear(in_features=128, out_features=2)
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
