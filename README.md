# AITM-pytorch

- [official Tensorflow implementation](https://github.com/xidongbo/AITM)
- [reference AITM-torch](https://github.com/adtalos/AITM-torch)

## DataSet && Preprocess

- [public Ali-CCP (Alibaba Click and Conversion Prediction) dataset](https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408)
```python
python preprocess_dataset.py
```

## Train && Test
```
python train.py &
tail -f log/train.log
```

## AUC performance
```
cilck auc: 0.6176967627520036; conversion auc: 0.6533093712496391
```
