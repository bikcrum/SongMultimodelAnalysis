import numpy as np
import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):

    def __init__(self, df, vocab, transform=None):
        super().__init__()

        self.df = df.copy()

        self.transform = transform

        self.vocab = vocab

        # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        self.specs = []
        self.lyrics = []

        for i, row in df.iterrows():
            spec = np.load(f'{row.file_path}')
            self.specs.append(spec)
            #
            # lyric = tokenizer(row.lyrics,
            #                   padding='max_length',
            #                   # 60 because of mean words count in lyrics
            #                   max_length=256,
            #                   truncation=True,
            #                   return_tensors="pt")
            #
            self.lyrics.append(torch.tensor(vocab.lookup_indices(row.lyrics.split(' '))))

        self.specs = np.array(self.specs)
        # self.specs = self.specs.squeeze()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        spec = self.specs[idx]
        lyric = self.lyrics[idx]

        spec = torch.Tensor(spec)

        if self.transform:
            spec = self.transform(spec)

        label = self.df.iloc[idx]['label']
        target = self.df.iloc[idx][['valence', 'arousal']].astype(np.float32)

        return self.df.iloc[idx].dzr_sng_id, label, spec, lyric, target
