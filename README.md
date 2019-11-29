# PointNet implementation practice
This repository aims at implementing [PointNet](https://arxiv.org/abs/1612.00593) in a simple way. The scripts are run on my personal laptop with a GTX-1050 4GB. The number of sampling points and batch size are less than the original implementation because of limited GPU resource. The testing point clouds are also randomly sampled, thus the accuracy serves as a reference of how the model works. (The original implementation inference all the points)

This repo supports training [ModelNet10/ModelNet40](https://modelnet.cs.princeton.edu/) dataset for classification, and [Stanford3D](http://buildingparser.stanford.edu/dataset.html) dataset for scene segmentation. Please refer to the original [git repository](https://github.com/charlesq34/pointnet) for more detailed implementation.

First, generate dataset and parse data into .npy files, then start training. Parameters can be modified in both training scripts.

## [ModelNet10](https://modelnet.cs.princeton.edu/)
Generate Dataset
```
python3 dataset/model_net_dataset.py -modelnet10 -p FOLDER_PATH
```
Start training
```
python3 train/train_modelnet.py -m10
```
Parameter used:
- batch size: 16
- learning rate: 1e-3
- weight decay: 1e-5
- number of sampled points: 1500
- epochs: 8
- w/o feature transformation

Accuracy: 86.12%

## [ModelNet40](https://modelnet.cs.princeton.edu/)
Generate Dataset
```
python3 dataset/model_net_dataset.py -modelnet40 -p FOLDER_PATH
```
Start training
```
python3 train/train_modelnet.py -m40
```
Parameter used:
- batch size: 24
- learning rate: 1e-3
- weight decay: 1e-4
- number of sampled points: 1024
- epochs: 10
- w/o feature transformation

Accuracy: 78.24%

## [Stanford3D](http://buildingparser.stanford.edu/dataset.html)
Generate Dataset
```
python3dataset/stanford_3d_dataset.py -p FOLDER_PATH
```
Start training
```
python3 train/train_stanford3d.py
```
Parameter used:
- batch size: 16
- learning rate: 1e-3
- weight decay: 1e-4
- number of sampled points: 2048
- epochs: 100
- learning rate decay: 0.5 for every 20 epochs
- w/o feature transformation

Accuracy: 60.85% (calculated on points)

## Reference
- [fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)
- [charlesq34/pointnet](https://github.com/charlesq34/pointnet)
