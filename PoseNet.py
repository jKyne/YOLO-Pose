import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DistanceNet(nn.Module):
    def __init__(self):
        super(DistanceNet, self).__init__()
        self.fc1 = nn.Linear(15, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 3)
        self.loss = DistanceLoss()

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.fc2(x), 0.1, inplace=True)
        x = self.fc3(x)
        return x

class RotationNet(nn.Module):
    def __init__(self):
        super(RotationNet, self).__init__()
        self.fc1 = nn.Linear(18, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)
        self.loss = RotationLoss()

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.fc2(x), 0.1, inplace=True)
        x = F.tanh(self.fc3(x))
        return x

class DistanceLoss(nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()
    
    def forward(self, output, ground_truth):
        loss = nn.MSELoss(size_average=True)(output, ground_truth)
        return loss

class RotationLoss(nn.Module):
    def __init__(self):
        super(RotationLoss, self).__init__()
        self.control_points = torch.FloatTensor([[[1,0,0]],[[0,1,0]],[[-1,0,0]],[[0,-1,0]]]).cuda()

    
    def forward(self, output_R, output_T, gt_R, gt_T):
        loss = nn.MSELoss(size_average=True)(output_R, gt_R)
        '''
        batch = output_R.size(0)
        gt_proj = torch.zeros(batch, 4, 3).cuda()
        pred_proj = torch.zeros(batch, 4, 3).cuda()
        loss = 0

        for j in range(batch):
            (x,y,z,w) = (gt_R[j][0],gt_R[j][1],gt_R[j][2],gt_R[j][3])
            gt_rotation =torch.Tensor([[2*(x**2+w**2)-1, 2*(x*y+z*w), 2*(x*z-y*w)],
                                        [2*(x*y-z*w), 2*(y**2+w**2)-1, 2*(y*z+x*w)],
                                        [2*(x*z+y*w), 2*(y*z-x*w), 2*(z**2+w**2)-1]])
            gt_rotation = gt_rotation.cuda()
            (x,y,z,w) = (output_R[j][0],output_R[j][1],output_R[j][2],output_R[j][3])
            pred_rotation =torch.Tensor([[2*(x**2+w**2)-1, 2*(x*y+z*w), 2*(x*z-y*w)],
                        [2*(x*y-z*w), 2*(y**2+w**2)-1, 2*(y*z+x*w)],
                        [2*(x*z+y*w), 2*(y*z-x*w), 2*(z**2+w**2)-1]])
            pred_rotation = pred_rotation.cuda()

            for i in range(4):
                gt_proj[j][i] = gt_T[j] + torch.mm(gt_rotation, self.control_points[i].t()).view(-1)
                pred_proj[j][i] = output_T[j] + torch.mm(pred_rotation,self.control_points[i].t()).view(-1)
            
            #
            loss_tmp = 1000
            loss = 0
            loss_tmp = torch.zeros(4,4).cuda()
            loss_m = torch.zeros(4).cuda()
            for m in range(4):
                for n in range(4):
                    loss_tmp[m][n] = nn.MSELoss(size_average=True)(pred_proj[j][m],gt_proj[j][n])
            loss_m, index = torch.min(loss_tmp,dim=1)
            loss += torch.mean(loss_m)
            #
            loss = 0
            loss_m = torch.zeros(4).cuda()
            for m in range(4):
                loss_m[m] = nn.MSELoss(size_average=True)(pred_proj[j][m],gt_proj[j][m])
            loss += torch.mean(loss_m)
        '''
        return loss