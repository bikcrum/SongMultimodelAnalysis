import os
import sys

import pandas as pd

from utils import AudioUtil, LyricsUtil


def main():
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else ''

    os.makedirs(os.path.join(dataset_dir, 'cache/specs'), exist_ok=True)

    df = pd.read_csv(os.path.join(dataset_dir, 'dataset.csv'))

    # Merge lyrics df
    df_lyrics = pd.read_csv(os.path.join(dataset_dir, 'lyrics.csv'))

    assert len(df) == len(df_lyrics)

    df = df.merge(df_lyrics, on='dzr_sng_id')

    # AUDIO PREPROCESSING
    if len(sys.argv) > 2 and 'a' == sys.argv[2]:
        print("Processing audio")
        # Add necessary source and target file path
        df['file_path'] = df.dzr_sng_id.apply(lambda song_id: os.path.join(dataset_dir, f'wav/{song_id}.wav'))
        df['save_file_path'] = df.dzr_sng_id.apply(
            lambda song_id: os.path.join(dataset_dir, f'cache/specs/{song_id}.npy'))

        # Convert wav to spec and save it in target file path
        AudioUtil.save_spectrogram(df)

    # LYRICS PREPROCESSING
    if len(sys.argv) > 2 and 'l' == sys.argv[2]:
        print("Processing lyrics")
        # Clean lyrics
        df_lyrics['lyrics'] = LyricsUtil.get_cleaned_lyrics(df_lyrics)

        # Save cleaned lyrics
        df_lyrics.to_csv(os.path.join(dataset_dir, 'cache/lyrics-cleaned.csv'), index=None)

    # LYRICS AUGUMENTATION
    if len(sys.argv) > 2 and 'laug' == sys.argv[2]:
        df = pd.read_csv(os.path.join(dataset_dir, 'cache/lyrics-cleaned.csv'))
        df_aug = LyricsUtil.back_translation(df, 'zh')
        df_aug.lyrics = LyricsUtil.get_cleaned_lyrics(df_aug)
        df_aug.to_csv(os.path.join(dataset_dir, 'cache/lyrics-augmented-zh.csv'), index=None)


if __name__ == '__main__':
    main()
