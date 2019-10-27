from __future__ import print_function
import torch
from torch.utils.data import Dataset, random_split
import os
import sys
import numpy as np
import random
import argparse
from tqdm import tqdm

class ModelNetDataset(Dataset):
    categories10 = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    categories40 = ['airplane','bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 
                    'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 
                    'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 
                    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

    def __init__(self, dataset_path, num_points=-1, mode='train', percentage=1.0):
        self.dataset_path = dataset_path + mode + '.txt'
        self.mode = mode
        self.num_points = num_points
        self.data_path = [] # [[data_path, category], ...]
        
        assert 0.0 < percentage <= 1.0
        
        self.data_path = [line.strip().split() for line in open(self.dataset_path, 'r').readlines()]
        
        # shrink dataset
        random.shuffle(self.data_path)
        self.data_path = self.data_path[:int(len(self.data_path) * percentage)]

    def __getitem__(self, index):
        points = np.load(self.data_path[index][0])

        # random sampling
        if self.num_points != -1:
            rand_idx = np.random.choice(points.shape[0], self.num_points, replace=True)
            points = points[rand_idx, :]

        # points scaling
        points -= np.mean(points, axis=0, keepdims=True)
        points /= np.max(np.linalg.norm(points, axis=1))

        return torch.from_numpy(points), torch.tensor(int(self.data_path[index][1]), dtype=torch.long)

    def __len__(self):
        return len(self.data_path)


def generate_model_net_dataset(root_dir, file_name, mode='test', categories=ModelNetDataset.categories10):
    if not os.path.exists('dataset/%s' %(file_name)):
        os.mkdir('dataset/%s' %(file_name))
    if not os.path.exists('dataset/%s/%s' %(file_name, mode)):
        os.mkdir('dataset/%s/%s' %(file_name, mode))
    if os.path.exists('dataset/%s/%s.txt' % (file_name, mode)):
        print('dataset/%s/%s.txt detected! Please remove the previous generated dataset!' %(file_name, mode))
        return

    data_path = []
    for index, cat in enumerate(categories):
        folder_path = os.path.join(root_dir, cat, mode)
        for point_path in os.listdir(folder_path):
            data_path.append((index, os.path.join(folder_path, point_path)))
    idx = 0
    with open('dataset/%s/%s.txt' % (file_name, mode), 'w') as f:
        for cat, path in tqdm(data_path):
            target_path = 'dataset/%s/%s/%s%06d.npy' % (file_name, mode, mode, idx)
            np.save(target_path, load_points(path))
            f.write('%s %d\n' % (target_path, cat))
            idx += 1
            

def load_points(path):

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
    parser = argparse.ArgumentParser(description='Dataset generation for ModelNet')
    parser.add_argument('-modelnet10', '--modelnet10', action='store_true', default=False, help='Generate ModelNet10 dataset')
    parser.add_argument('-modelnet40', '--modelnet40', action='store_true', default=False, help='Generate ModelNet40 dataset')
    parser.add_argument('-p', '--path', action='store', type=str, help='Directory to ModelNet')
    args = parser.parse_args()

    if args.modelnet10 and args.modelnet40:
        print('Cannot generate both dataset at same time!')
        exit(1)
    
    if not args.modelnet10 and not args.modelnet40:
        print('Please select which dataset to generate with flag -modelnet10 or -modelnet40')
        exit(1)

    if args.modelnet10:
        print('Generating ModelNet10 training set.')
        generate_model_net_dataset(args.path, 'modelnet10', mode='train')
        print('Generating ModelNet10 testing set.')
        generate_model_net_dataset(args.path, 'modelnet10', mode='test')

    if args.modelnet40:
        print('Generating ModelNet40 training set.')
        generate_model_net_dataset(args.path, 'modelnet40', mode='train', categories=ModelNetDataset.categories40)
        print('Generating ModelNet40 testing set.')
        generate_model_net_dataset(args.path, 'modelnet40', mode='test', categories=ModelNetDataset.categories40)
