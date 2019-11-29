import torch
import torch.nn as nn
from model.tnet import TNet

class PointNetSegmentation(nn.Module):
    '''
    PointNet for segmentation \n
    Input: batch_size x num_points x points_dim \n
    Output: batch_size x num_points x num_class
    '''

    def __init__(self, num_class, points_dim=9, feature_transformation=False):
        super(PointNetSegmentation, self).__init__()
        self.num_class = num_class
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
        self.conv6 = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512)
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256)
        )
        self.conv8 = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128)
        )
        self.conv9 = nn.Conv1d(128, num_class, 1)

    def forward(self, x):
        trans_mtx1 = self.trans1(x.transpose(1, 2))
        x = torch.bmm(x, trans_mtx1).transpose(1, 2)
        point_feature = self.conv2(x)
        if self.feature_transformation:
            trans_mtx2 = self.trans3(point_feature)
            point_feature = torch.bmm(point_feature.transpose(1, 2), trans_mtx2).transpose(1, 2)
        else:
            trans_mtx2 = None
        x = self.conv4(point_feature)
        x = self.conv5(x)
        global_feature = torch.max(x, dim=2, keepdim=True)[0].repeat(1, 1, x.shape[2])
        
        # concatenate global and point feature and run the segmentation network
        x = torch.cat([point_feature, global_feature], dim=1)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = x.transpose(2,1).contiguous()

        return x, trans_mtx2