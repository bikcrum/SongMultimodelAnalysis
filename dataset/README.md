### Dataset compiled from Deezer API

Link: https://drive.google.com/drive/folders/1DTgppYHu2V8naG2FOda0K-nkPCMbpbJM?usp=sharing

#### Notes:
1. Melspectrogram is converted from audio file in following configuration. Note that silence of 0-2 seconds is added to make all audio of same duration. 

Name|Value
---|---
FFT_SIZE | 1024
HOP_SIZE | 512
N_MEL_BINS | 64
N_SPECTROGRAM_BINS | FFT_SIZE // 2 + 1
F_MIN | 0.0
F_MAX | 22050 / 2
SAMPLE_RATE | 22050