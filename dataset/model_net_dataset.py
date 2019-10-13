from __future__ import print_function
import torch
from torch.utils.data import Dataset, random_split
import os
import sys
import numpy as np
import random

class ModelNetDataset(Dataset):
    categories10 = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    categories40 = ['airplane','bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 
                    'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 
                    'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 
                    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

    def __init__(self, root_dir, num_points=1500, mode='train', num_cat=10, percentage=1.0):
        self.root_dir = root_dir
        self.mode = mode
        self.num_points = num_points
        self.data_path = [] # [(category, data_path), ...]
        if num_cat == 10:
            self.categories = self.categories10
        elif num_cat == 40:
            self.categories = self.categories40
        else:
            raise NotImplementedError
        
        assert 0.0 < percentage <= 1.0
        
        # load data path
        for index, cat in enumerate(self.categories):
            folder_path = os.path.join(root_dir, cat, mode)
            for point_path in os.listdir(folder_path):
                self.data_path.append((index, os.path.join(folder_path, point_path)))
        
        # shrink dataset
        random.shuffle(self.data_path)
        self.data_path = self.data_path[:int(len(self.data_path) * percentage)]

    def __getitem__(self, index):
        points = self._load_point_clouds(self.data_path[index][1])

        # random sampling
        rand_idx = np.random.choice(points.shape[0], self.num_points, replace=True)
        points = points[rand_idx, :]

        # points scaling
        points -= np.mean(points, axis=0, keepdims=True)
        points /= np.max(np.linalg.norm(points, axis=1))

        return torch.from_numpy(points), torch.tensor(self.data_path[index][0], dtype=torch.long)

    def __len__(self):
        return len(self.data_path)
    
    def _load_point_clouds(self, path):

        with open(path, 'r') as f:
            data = f.readlines()
            data = [s.strip() for s in data]

        if data[0] == 'OFF':
            data.pop(0)
        elif data[0].find('OFF') != -1:
            data[0] = data[0][3:]
        else:
            print('Wrong data format!')
            raise Exception()

        num_vertices = int(data[0].split()[0])
        vertices = []
        for i in range(num_vertices):
            vertex = data[i+1].split()
            vertices.append([float(point) for point in vertex])

        return np.array(vertices)


if __name__ == "__main__":
    dataset = ModelNetDataset('/home/chikuan/Documents/ModelNet10/ModelNet10')
    val_size = int(len(dataset)*0.2)
    d1, d2 = random_split(dataset, [len(dataset) - val_size, val_size])
    print(len(d1), len(d2))
    print(len(dataset))
    print(dataset[0][0].shape, dataset[0][1])
    # pcd = open3d.PointCloud()
    # pcd.points = open3d.Vector3dVector(dataset[0][0])
    # open3d.draw_geometries([pcd])