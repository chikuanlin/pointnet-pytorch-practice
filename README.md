# PointNet implementation practice
This repository aims at implementing [PointNet](https://arxiv.org/abs/1612.00593) in a simple way. It supports training [ModelNet10/ModelNet40](https://modelnet.cs.princeton.edu/) dataset for classification, and [Stanford3D](http://buildingparser.stanford.edu/dataset.html) dataset for segmentation. Please refer to the original [git repository](https://github.com/charlesq34/pointnet) for more detailed implementation.

First, generate dataset and parse data into .npy files, then start training. Parameters can be modified in both train scripts.

## ModelNet10
```
python3 dataset/model_net_dataset.py -modelnet10 -p FOLDER_PATH
python3 train/train_modelnet.py -m10
```
## ModelNet40
```
python3 dataset/model_net_dataset.py -modelnet40 -p FOLDER_PATH
python3 train/train_modelnet.py -m40
```
## Stanford3D
```
python3dataset/stanford_3d_dataset.py -p FOLDER_PATH
python3 train/train_stanford3d.py
```
