import torch
import torch.nn as nn

class TNetKxK(nn.Module):
    '''
    Input: batch_size x K x num_points \n
    Output: batch_size x K x K
    '''
    def __init__(self, k=3):
        super(TNetKxK, self).__init__()
        self.k = k
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc5 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc6 = nn.Linear(256, k*k)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.max(x, dim=2)[0]
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        trans_mtx = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k * self.k)

        return (x + trans_mtx).view(-1, self.k, self.k)