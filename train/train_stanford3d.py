import os
import sys
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split 
import torch
import torch.nn as nn
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from dataset.stanford_3d_dataset import Stanford3dDetaset
from model.pointnet_segmentation import PointNetSegmentation
from model.tnet import tnet_regularization


def train_epoch(train_loader, net, criterion, optimizer, device, ft_reg):
    net = net.train()
    train_loss = 0.0
    train_acc = 0.0
    for x, labels in tqdm(train_loader):
        x = x.to(device)
        labels = labels.to(device)
        output, feat_mtx = net(x)
        output = output.view(-1, output.shape[-1])
        labels = labels.view(-1)
        loss = criterion(output, labels) + tnet_regularization(feat_mtx) * ft_reg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += check_accuracy(output, labels) / labels.shape[0]
        optimizer.zero_grad()
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    return train_loss, train_acc


def test_epoch(test_loader, net, criterion, device, ft_reg):
    net = net.eval()
    test_loss = 0.0
    test_acc = 0.0
    num_points = 0
    with torch.no_grad():
        for x, labels in tqdm(test_loader):
            x = x.to(device)
            labels = labels.to(device)
            output, feat_mtx = net(x)
            output = output.view(-1, output.shape[-1])
            labels = labels.view(-1)
            loss = criterion(output, labels) + tnet_regularization(feat_mtx) * ft_reg
            test_loss += loss.item()
            test_acc += check_accuracy(output, labels)
            num_points += labels.shape[0]
        test_loss /= len(test_loader)
        test_acc /= num_points
    return test_loss, test_acc


def check_accuracy(output, labels):
    return torch.sum((torch.max(output, dim=1)[1] == labels).float()).item()


def main(args):
    
    test_area = [6]
    if args.use_val:
        train_areas = [1, 2, 3, 5]
        val_area = [4]
        val_dataset = Stanford3dDetaset(areas=val_area, num_points=args.num_points)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    else:
        train_areas = [1, 2, 3, 4, 5]

    train_dataset = Stanford3dDetaset(areas=train_areas, num_points=args.num_points)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    net = PointNetSegmentation(13, points_dim=9, feature_transformation=args.feature_transform).double().to(device)

    if args.model_path is not None:
        net.load_state_dict(torch.load(args.model_path))
        model_path = args.model_path
        print('Model loaded from ', args.model_path)
    else:
        model_path = time.strftime("model_%H_%M_%S.pth", time.gmtime())
        print('New model path: %s' %(model_path))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    print('Start training:')
    for epoch in range(args.epochs):
        tic = time.time()

        train_loss, train_acc = train_epoch(train_loader, net, criterion, optimizer, device, args.feature_transform_reg)
        scheduler.step()

        val_loss, val_acc = 0.0, 0.0
        if args.use_val and epoch % 3 == 0:
            val_loss, val_acc = test_epoch(val_loader, net, criterion, device, args.feature_transform_reg)
        
        torch.save(net.state_dict(), model_path)
        print('Epoch [%3d/%3d] train loss: %.6f train acc: %.4f val loss: %.6f val acc: %.4f elasped time(s) %.2f' %(
            epoch+1, args.epochs, train_loss, train_acc, val_loss, val_acc, time.time()-tic))

    print('Start Testing:')
    test_dataset = Stanford3dDetaset(areas=test_area, num_points=args.num_points)# num_points=-1
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_loss, test_acc = test_epoch(test_loader, net, criterion, device, args.feature_transform_reg)
    print('Test result: test loss %.6f test acc: %.4f' %(test_loss, test_acc))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Stanford3d')
    parser.add_argument('-n', '--num_points', action='store', type=int, default=2048)
    parser.add_argument('-p', '--model_path', action='store', type=str, default=None)
    parser.add_argument('-ft', '--feature_transform', action='store_true', default=False)
    parser.add_argument('-ftreg', '--feature_transform_reg', action='store', type=float, default=1e-3)
    parser.add_argument('-lr', '--learning_rate', action='store', type=float, default=1e-3)
    parser.add_argument('-w', '--weight_decay', action='store', type=float, default=1e-4)
    parser.add_argument('-b', '--batch_size', action='store', type=int, default=16)
    parser.add_argument('-e', '--epochs', action='store', type=int, default=100)
    parser.add_argument('-val', '--use_val', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
