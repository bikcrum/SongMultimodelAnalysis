import os

import pandas as pd
import ray
import requests
from tqdm import tqdm
import json
from bs4 import BeautifulSoup
import urllib3
import shutil

ray.init(ignore_reinit_error=True)


@ray.remote
def fetch_batch_song_infos(df, start, end):
    print(f'Batch: {start}-{end}')
    out = []
    for i in range(start, end):
        r = requests.get(f'https://api.deezer.com/track/{df.iloc[i].dzr_sng_id}')
        out.append(r.json())
    return out


final_song_infos = []


def fetch_song_infos(df):
    if len(df) == 0:
        return

    global final_song_infos

    batch_size = 30
    batch_refs = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch_refs.append(fetch_batch_song_infos(df, i, min(len(df), i + batch_size)))

    # wait for batches to finish
    # song_infos = ray.get(batch_refs)

    # flatten
    song_infos = sum(song_infos, [])

    # get info of songs that is fetched
    song_infos = list(filter(lambda x: 'id' in x.keys(), song_infos))

    final_song_infos += song_infos

    # get only ids
    song_ids = list(map(lambda song_info: song_info['id'], song_infos))

    # get ids of songs that needs to be re-fetched
    _df = df[~df.dzr_sng_id.isin(song_ids)]

    with open('song_infos.json', 'w') as p:
        json.dump(final_song_infos, p)

    _df.to_csv('remaining.csv')

    print(f'{len(song_infos)} songs fetched. {len(_df)} songs failed to fetch. Retrying...')

    # fetch remaining songs
    fetch_song_infos(_df)


final_song_previews = []


@ray.remote
def download_batch_song(song_infos, batch_id):
    print(f'Batch:{batch_id}')
    out = []
    for song_info in song_infos:
        id = song_info.get('id')
        title = song_info.get('title')
        artist_name = song_info.get('artist', {}).get('name')
        duration = song_info.get('duration')

        r = requests.get(f'https://api.deezer.com/search',
                         params={'q': "%s,%s" % (artist_name, title)})

        try:
            results = r.json().get('data', [])
        except Exception as e:
            print(id, e)
            continue

        # found = False
        for result in results:
            if result.get('title', '').lower() == title.lower() and result.get('artist', {}).get(
                    'name', '').lower() == artist_name.lower():
                out.append({'id': id, 'new_id': result.get('id'), 'preview': result.get('preview')})
                # found = True
                break
        #
        # if not found:
        #     out.append({'id': id,'results':results})
    return out


def download_songs(song_infos):
    if len(song_infos) == 0:
        return

    global final_song_previews

    batch_size = 50
    batch_refs = []
    for i in range(0, len(song_infos), batch_size):
        batch_refs.append(download_batch_song.remote(song_infos[i:i + batch_size], i))

    # wait for batches to finish
    song_previews = ray.get(batch_refs)

    # flatten
    song_previews = sum(song_previews, [])

    final_song_previews += song_previews

    # save song_previews
    with open('song_previews.json', 'w') as p:
        json.dump(final_song_previews, p)

    # get failed song_previews ids
    ids = set(map(lambda x: x['id'], song_infos))
    new_ids = set(map(lambda x: x['id'], song_previews))
    failed_ids = ids - new_ids

    # get song infos for failed song ids
    song_infos = list(filter(lambda song_info: song_info['id'] in failed_ids, song_infos))

    print(f'{len(song_previews)} song previews fetched. {len(failed_ids)} song previews failed to fetch. Retrying...')

    # download failed song
    download_songs(song_infos)


final_song_lyrics = []


@ray.remote
def fetch_batch_lyrics(song_previews, batch_id):
    print(f'Batch:{batch_id}')
    out = []
    for song_preview in song_previews:
        id = song_preview.get('new_id')

        r = requests.get(f'https://www.deezer.com/us/track/{id}')

        bs = BeautifulSoup(r.text, features="lxml")
        divs = bs.find_all('div')
        div_lyrics_label = list(filter(lambda div: div[1].text == 'Lyrics', enumerate(divs)))

        # Check if there is lyrics
        if len(div_lyrics_label) > 0:
            div_lyrics = divs[div_lyrics_label[0][0] + 1].text
            out.append({'new_id': id, 'lyrics': div_lyrics})

    return out


def fetch_lyrics(song_previews):
    if len(song_previews) == 0:
        return

    global final_song_lyrics

    batch_size = 30
    batch_refs = []
    for i in range(0, len(song_previews), batch_size):
        batch_refs.append(fetch_batch_lyrics.remote(song_previews[i:i + batch_size], i))

    # wait for batches to finish
    song_lyrics = ray.get(batch_refs)

    # flatten
    song_lyrics = sum(song_lyrics, [])

    final_song_lyrics += song_lyrics

    # save songs
    with open('songs_lyrics.json', 'w') as p:
        json.dump(final_song_lyrics, p)

    # get failed songs ids
    ids = set(map(lambda x: x['new_id'], song_previews))
    new_ids = set(map(lambda x: x['new_id'], song_lyrics))
    failed_ids = ids - new_ids

    # get song infos for failed song ids
    song_previews = list(filter(lambda song_preview: song_preview['new_id'] in failed_ids, song_previews))

    print(f'{len(song_lyrics)} song lyrics fetched. {len(failed_ids)} song lyrics failed to fetch. Retrying...')

    # download failed song
    fetch_lyrics(song_previews)


@ray.remote
def download_batch_mp3(song_previews, batch_id):
    print(f'Batch:{batch_id}-{batch_id + len(song_previews)}')
    for i, song_preview in song_previews.iterrows():
        http = urllib3.PoolManager()
        with open(f"deezer_mood_detection_dataset-master/previews/{song_preview.dzr_sng_id}.mp3", 'wb') as out:
            r = http.request('GET', song_preview.preview, preload_content=False)
            shutil.copyfileobj(r, out)


def download_mp3(song_previews):
    if len(song_previews) == 0:
        return

    batch_size = 100
    refs = []
    for i in range(0, len(song_previews), batch_size):
        refs.append(download_batch_mp3.remote(song_previews[i:min(len(song_previews), i + batch_size)], i))

    # download batch songs
    ray.get(refs)

    # get downloads song ids
    ids = list(map(lambda file: int(file.split('.')[0]), os.listdir('deezer_mood_detection_dataset-master/previews')))

    # get remaining song ids
    rids = set(song_previews.dzr_sng_id) - set(ids)

    print(f'{len(ids)} songs mp3 download. {len(rids)} failed to download. Retrying...')

    # download remaining songs
    download_mp3(song_previews[song_previews.dzr_sng_id.isin(rids)])


def main():
    df = pd.read_csv('deezer_mood_detection_dataset-master/test.csv')
    fetch_song_infos(df)

    with open('deezer_mood_detection_dataset-master/test_song_infos.json') as song_infos:
        download_songs(json.load(song_infos))

    with open('deezer_mood_detection_dataset-master/test_song_previews.json') as song_previews:
        fetch_lyrics(json.load(song_previews))

    df = pd.read_csv('deezer_mood_detection_dataset-master/previews.csv')
    download_mp3(df)

    print('Finished')


if __name__ == '__main__':
    main()
