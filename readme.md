# readme
detrを使った物体認識のサンプル

参考
https://huggingface.co/facebook/detr-resnet-50

## install
`pip install -r requirements.txt`

```
$ python --version
Python 3.9.5
```

### run

物体認識のサンプル  
`python sample.py`


サイズ推定のサンプル
`python main.py`


loadするimageはこのあたり書き換えてください
`image = Image.open("./images/ninjin.jpeg")`


