### SongMultimodelAnalysis - Deep learning project (AI-535)

## Get started

1. Download scripts from here https://drive.google.com/drive/folders/1G7fdJRXJj9w0n6hogd01xn2Wrj6mgqAd?usp=sharing

Order| Script                          |Decription
---|---------------------------------|---
1.| create_empty_dataset_folders.sh |This creates empty directory where datasets will be downloaded. **You must change working directory in this file** but you should not following dataset path.
2.| download_mp3.sh                 | Your mp3 songs will be downloaded              
3.| converter_wav.sh                | Converts all your mp3 songs to wav format      
4.| converter_melspectrogram.sh     | Converts all your wav songs to melspectrograms 
5.| train.sh | Trains your model

2. You must follow the order of execution as given above. For example. `converter_wav.sh` cannot be run before `download_mp3.sh`

#### Important note:
1. Melspectrogram is converted from wav file in following configuration. Note that silence of 0-2 seconds is added to make all audio of same duration. 

Name|Value
---|---
fft size | 1024
hop size | 512
mel bins count | 64
spectrogram bins count | FFT_SIZE // 2 + 1
fmin | 0.0
fmax | 22050 // 2
sample rate | 22050