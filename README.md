# Deep Amp

Guitar Amp and Effect simulator with Deep Learning technology

## how to setup

# Training Dataset and Test dataset 
Prepare for two kind of audio files

1. Prepare for dry guitar audio file without any effect, wav format, 16 bit, 48 kHz monoral

1. Prepare for wet guitar audio with guitar or overdrive effect, wav format, 16 bit, 48 kHz monoral

1. Update config.yml with your file names


# train model
```sh
python train.py -c config.yml
```

# predict by trained model
```sh
python predict.py -c config.yml -i input.wav -o predicted.wav -m checkpoint/20180208_235128/model_000031.h5
```
