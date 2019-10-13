import os
import sys
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split 
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from dataset.model_net_dataset import ModelNetDataset
from model.pointnet_classification import PointNetClassification

params = {
    # Model
    'num_points': 1500,
    'model_path': None,
    'num_class': 10,
    'dataset_path': '/home/chikuan/Documents/ModelNet10/ModelNet10',
    'dataset_percentage': 0.2,
    'feature_transformation': False,

    # Training
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'batch_size': 16,
    'epochs': 5
}

def train_epoch(train_loader, net, criterion, optimizer, device):
    net = net.train()
    train_loss = 0.0
    train_acc = 0.0
    for x, labels in tqdm(train_loader):
        x = x.to(device)
        labels = labels.to(device)
        output = net(x)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += check_accuracy(output, labels)
        optimizer.zero_grad()
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    return train_loss, train_acc


def test_epoch(test_loader, net, criterion, device):
    net = net.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for x, labels in tqdm(test_loader):
            x = x.to(device)
            labels = labels.to(device)
            output = net(x)
            loss = criterion(output, labels)
            test_loss += loss.item()
            test_acc += check_accuracy(output, labels)
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
    return test_loss, test_acc


def check_accuracy(output, labels):
    return torch.mean((torch.max(output, dim=1)[1] == labels).float()).item()


def main():
    train_dataset = ModelNetDataset(params['dataset_path'], 
                                    num_points=params['num_points'], 
                                    mode='train', 
                                    num_cat=params['num_class'],
                                    percentage=params['dataset_percentage'])
    val_length = int(len(train_dataset) * 0.15)
    train_dataset, val_dataset = random_split(train_dataset, [len(train_dataset) - val_length, val_length])
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = PointNetClassification(num_class=params['num_class'], 
                                 feature_transformation=params['feature_transformation']
                                ).double().to(device)

    if params['model_path'] is not None:
        net.load_state_dict(torch.load(params['model_path']))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    print('Start training:')
    for epoch in range(params['epochs']):
        tic = time.time()

        train_loss, train_acc = train_epoch(train_loader, net, criterion, optimizer, device)
        val_loss, val_acc = test_epoch(val_loader, net, criterion, device)

        print('Epoch [%3d/%3d] train loss: %.6f train acc: %.4f val loss: %.6f val acc: %.4f elasped time(s) %.2f' %(
            epoch+1, params['epochs'], train_loss, train_acc, val_loss, val_acc, time.time()-tic))
    
    if params['model_path'] is not None:
        model_path = params['model_path']
    else:
        model_path = 'model.pth'
    print('Saving model to: %s' %(model_path))
    torch.save(net.state_dict(), model_path)

    print('Start Testing:')
    test_dataset = ModelNetDataset(params['dataset_path'], 
                                   num_points=params['num_points'], 
                                   mode='test', 
                                   num_cat=params['num_class'],
                                   percentage=params['dataset_percentage'])
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loss, test_acc = test_epoch(test_loader, net, criterion, device)
    print('Test result: test loss %.6f test acc: %.4f' %(test_loss, test_acc))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
    