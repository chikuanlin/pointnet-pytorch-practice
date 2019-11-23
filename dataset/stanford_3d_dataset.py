from __future__ import print_function
import torch
from torch.utils.data import Dataset, random_split
import os
import sys
import numpy as np
import random
from tqdm import tqdm
import argparse

class Stanford3dDetaset(Dataset):
    categories = {
        'ceiling':0, 'floor':1, 'wall':2, 'beam':3, 'column':4, 'window':5, 'door':6, 
        'table':7, 'chair':8, 'sofa':9, 'bookcase':10, 'board':11, 'clutter':12
    }

    def __init__(self, areas=[], num_points=-1):
        self.data_path = []
        self.num_points = num_points
        for area in areas:
            path = 'dataset/stanford3d/Area_%d.txt' %(area)
            self.data_path += [line.rstrip('\n') for line in open(path, 'r').readlines()]

    def __getitem__(self, index):
        points = np.load(self.data_path[index])
        labels = points[:, -1]
        points = points[:, :-1]

        # random sampling
        if self.num_points != -1:
            rand_idx = np.random.choice(points.shape[0], self.num_points, replace=True)
            points = points[rand_idx, :]
            labels = labels[rand_idx]

        # points scaling
        points[:, :3] -= np.mean(points[:, :3], axis=0, keepdims=True)
        points[:, :3] /= np.max(np.linalg.norm(points[:, :3], axis=1))
        points[:, 3:] /= 255.0

        return torch.from_numpy(points), torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data_path)

def generate_stanford_3d_dataset(root_dir):
    if not os.path.exists('dataset/stanford3d'):
        os.mkdir('dataset/stanford3d')
    areas = [f for f in os.listdir(root_dir) if not f.startswith('.')]
    for area in areas:
        if not os.path.exists('dataset/stanford3d/%s' %(area)):
            os.mkdir('dataset/stanford3d/%s' %(area))
        area_path = os.path.join(root_dir, area)
        area_text = ''
        rooms = [f for f in os.listdir(area_path) if not f.startswith('.')]
        print('Generating dataset for %s' %(area))
        for room in tqdm(rooms):
            room_path = os.path.join(area_path, room, 'Annotations')
            instances = [f for f in os.listdir(room_path) if not f.startswith('.')]
            room_points = []
            room_cats = []
            for instance in instances:
                instance_path = os.path.join(room_path, instance)
                cat = instance.split('_')[0]
                if cat in Stanford3dDetaset.categories:
                    instance_points = load_points(instance_path)
                    labels = np.repeat(Stanford3dDetaset.categories[cat], instance_points.shape[0])
                    room_points.append(instance_points)
                    room_cats.append(labels)
            room_points = np.concatenate(room_points, axis=0)
            room_cats = np.concatenate(room_cats)
            data = np.concatenate([room_points, room_cats[:, np.newaxis]], axis=1)
            data_path = 'dataset/stanford3d/%s/%s.npy' %(area, room)
            area_text += data_path + '\n'
            np.save(data_path, data)
        with open('dataset/stanford3d/%s.txt' %(area), 'w') as f:
            f.write(area_text)
    

def load_points(path):
    data = [line.rstrip('\n').split() for line in open(path, 'r').readlines()]
    points = []
    for point_feature in data:
        try:
            feature = [float(point) for point in point_feature]
            if len(feature) == 6:
                points.append(feature)
        except:
            pass
    return np.array(points)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset generation for ModelNet')
    parser.add_argument('-p', '--path', action='store', type=str, help='Directory to Stanford3D')
    args = parser.parse_args()

    print('Generating Stanford3D dataset.')
    generate_stanford_3d_dataset(args.path)
