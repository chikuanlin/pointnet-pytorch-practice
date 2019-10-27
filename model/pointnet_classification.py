import torch
import torch.nn as nn
from model.tnet import TNet

class PointNetClassification(nn.Module):
    '''
    PointNet for classification \n
    Input: batch_size x num_points x points_dim \n
    Output: batch_size x num_class
    '''

    def __init__(self, num_class, points_dim=3, feature_transformation=False):
        super(PointNetClassification, self).__init__()
        self.feature_transformation = feature_transformation
        self.trans1 = TNet(k=points_dim)
        self.conv2 = nn.Sequential(
            nn.Conv1d(points_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        if self.feature_transformation:
            self.trans3 = TNet(k=64)
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024)
        )
        self.fc6 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc7 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc8 = nn.Linear(256, num_class)

    def forward(self, x):
        trans_mtx1 = self.trans1(x.transpose(1, 2))
        x = torch.bmm(x, trans_mtx1).transpose(1, 2)
        x = self.conv2(x)
        if self.feature_transformation:
            trans_mtx2 = self.trans3(x)
            x = torch.bmm(x.transpose(1, 2), trans_mtx2).transpose(1, 2)
        else:
            trans_mtx2 = None
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.max(x, dim=2)[0]
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        
        return x, trans_mtx2