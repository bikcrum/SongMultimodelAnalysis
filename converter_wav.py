import os
import sys

import pandas as pd
import ray

ray.init(ignore_reinit_error=True)

workdir = sys.argv[1] if len(sys.argv) > 1 else ''


# @ray.remote
def convert_batch_to_wav(dataset, batch_id):
    print(f'Batch:{batch_id}-{batch_id + len(dataset)}')
    for i, row in dataset.iterrows():
        command = f"ffmpeg -y -i {os.path.join(workdir, f'dataset/previews/mp3/{row.dzr_sng_id}.mp3')} -acodec pcm_u8 -ar 22050 {os.path.join(workdir, f'dataset/previews/wav/{row.dzr_sng_id}.wav')}"
        os.system(command)


def convert_to_wav(dataset):
    if len(dataset) == 0:
        return

    batch_size = 1
    refs = []
    for i in range(0, len(dataset), batch_size):
        refs.append(convert_batch_to_wav(dataset[i:min(len(dataset), i + batch_size)], i))

    # download batch songs
    # ray.get(refs)

    # get downloads song ids
    ids = list(map(lambda file: int(file.split('.')[0]), os.listdir(os.path.join(workdir, 'dataset/previews/wav'))))

    # get remaining song ids
    rids = set(dataset.dzr_sng_id) - set(ids)

    print(f'{len(ids)} songs converted to wav. {len(rids)} failed to convert. Retrying...')

    # download remaining songs
    convert_to_wav(dataset[dataset.dzr_sng_id.isin(rids)])


def main():
    df = pd.read_csv(os.path.join(workdir, 'dataset/dataset.csv'))
    convert_to_wav(df)

    print('Finished')


if __name__ == '__main__':
    main()
