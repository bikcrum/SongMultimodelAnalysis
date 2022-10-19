### SongMultimodelAnalysis - Deep learning project (AI-535)

### Project paper
https://github.com/bikcrum/SongMultimodelAnalysis/blob/master/Report.pdf

## Get started

1. Download scripts from here https://drive.google.com/drive/folders/1G7fdJRXJj9w0n6hogd01xn2Wrj6mgqAd?usp=sharing

Order| Script                         |Decription
---|--------------------------------|---
1.| create_empty_dataset_folders.sh |This creates empty directory where datasets will be downloaded. **You must change working directory in this file** but you should not change the following dataset path.
2.| download_mp3.sh                | Your mp3 songs will be downloaded              
3.| converter_wav.sh               | Converts all your mp3 songs to wav format      
~~4.~~ | ~~converter_melspectrogram.sh~~ | ~~Converts all your wav songs to melspectrograms. No need to do this. Data is now cached automatically during training.~~
4. | feature_extraction.sh          | Converts all wav song to melspectrogram and clean lyrics and saves in cache directory. [OPTIONS: al, a or l, where a=audio, l=lyrics]
5.| train.sh                       | Trains your model. You can view your tensorboard at http://localhost:6006

2. You must follow the order of execution as given above. For example. `converter_wav.sh` cannot be run before `download_mp3.sh`

#### Important note:
1. Melspectrogram is converted from wav file in configuration specified in utils.py file.
