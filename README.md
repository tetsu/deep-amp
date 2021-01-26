# Deep Amp

Guitar Amp and Effect simulator with Deep Learning technology

## how to setup

```sh
# train model
python train.py -c config.yml

# predict by trained model
python predict.py -c config.yml -i input.wav -o predicted.wav -m checkpoint/20180208_235128/model_000031.h5
```
