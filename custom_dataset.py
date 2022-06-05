import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer


class AudioDataset(Dataset):

    def __init__(self, df, transform=None):
        super().__init__()

        self.df = df.copy()

        self.transform = transform

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        self.specs = []
        self.lyrics = []

        for i, row in df.iterrows():
            spec = np.load(f'{row.file_path}')
            self.specs.append(spec)

            lyric = tokenizer(row.lyrics,
                              padding='max_length',
                              # 60 because of mean words count in lyrics
                              max_length=60,
                              truncation=True,
                              return_tensors="pt")

            self.lyrics.append(lyric)

        self.specs = np.array(self.specs)

        self.specs = self.specs.squeeze()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        spec = self.specs[idx]
        lyric = self.lyrics[idx]

        spec = torch.Tensor(spec)

        if self.transform:
            spec = self.transform(spec)

        class_label = self.df.iloc[idx]['label']

        return spec, lyric, class_label
