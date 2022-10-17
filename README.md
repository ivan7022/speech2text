# ASR project barebones

## Installation guide

1. Build and start docker container.
```shell
   docker build -t speech2text .
   docker run \
   --gpus '"device=0"' \
   -it --rm speech2text
```

2. Download best model checkpoint from https://disk.yandex.ru/d/nrjR_EMvW5wfSA or with script.
```shell
wget  -O model_best.pth https://file.io/X8k7zGWAg8y5
```

3. Download LM checkpoint.
```shell
wget https://www.openslr.org/resources/11/4-gram.arpa.gz --no-check-certificate
```

4. Inference model!
```shell
   python test.py -c hw_asr/configs/test_config.json -t test_data -b 5 -r model_best.pth -o output_other.json
```
or 
```shell
  python test.py -c hw_asr/configs/test_config.json -b 5 -r model_best.pth -o output_other.json
```
which runs eval on test-other by default.

5. Train?
```shell
   python train.py -c hw_asr/config.json
```
