import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *

def build_targets(pred_corners, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, seen):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    anchor_step = len(anchors)//num_anchors
    conf_mask   = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask  = torch.zeros(nB, nA, nH, nW)
    cls_mask    = torch.zeros(nB, nA, nH, nW)
    tx0         = torch.zeros(nB, nA, nH, nW) 
    ty0         = torch.zeros(nB, nA, nH, nW) 
    tx1         = torch.zeros(nB, nA, nH, nW) 
    ty1         = torch.zeros(nB, nA, nH, nW) 
    tx2         = torch.zeros(nB, nA, nH, nW) 
    ty2         = torch.zeros(nB, nA, nH, nW) 
    tx3         = torch.zeros(nB, nA, nH, nW) 
    ty3         = torch.zeros(nB, nA, nH, nW) 
    tx4         = torch.zeros(nB, nA, nH, nW) 
    ty4         = torch.zeros(nB, nA, nH, nW) 
    tx5         = torch.zeros(nB, nA, nH, nW) 
    ty5         = torch.zeros(nB, nA, nH, nW) 
    tconf       = torch.zeros(nB, nA, nH, nW)
    tcls        = torch.zeros(nB, nA, nH, nW) 

    nAnchors = nA*nH*nW
    nPixels  = nH*nW
    for b in range(nB):
        cur_pred_corners = pred_corners[b*nAnchors:(b+1)*nAnchors].t()
        cur_confs = torch.zeros(nAnchors).view(nA, nH, nW)
        for t in range(50):
            if target[b][t*15+1] == 0:
                break
            # g or gt refer to groundtruth
            gx0 = target[b][t*15+1]*nW
            gy0 = target[b][t*15+2]*nH
            gx1 = target[b][t*15+3]*nW
            gy1 = target[b][t*15+4]*nH
            gx2 = target[b][t*15+5]*nW
            gy2 = target[b][t*15+6]*nH
            gx3 = target[b][t*15+7]*nW
            gy3 = target[b][t*15+8]*nH
            gx4 = target[b][t*15+9]*nW
            gy4 = target[b][t*15+10]*nH
            gx5 = target[b][t*15+11]*nW
            gy5 = target[b][t*15+12]*nH

            cur_gt_corners = torch.FloatTensor([gx0/nW,gy0/nH,gx1/nW,gy1/nH,gx2/nW,gy2/nH,gx3/nW,gy3/nH,gx4/nW,gy4/nH,gx5/nW,gy5/nH]).repeat(nAnchors,1).t() # 16 x nAnchors
            cur_confs  = torch.max(cur_confs, corner_confidences9(cur_pred_corners, cur_gt_corners).view(nA, nH, nW)).view(nA, nH, nW) # some irrelevant areas are filtered, in the same grid multiple anchor boxes might exceed the threshold
        tmp_mask = (cur_confs < sil_thresh).float()
        conf_mask[b] = tmp_mask.view(nA, nH, nW)


    if seen < -1:#6400:
       tx0.fill_(0.5)
       ty0.fill_(0.5)
       tx1.fill_(0.5)
       ty1.fill_(0.5)
       tx2.fill_(0.5)
       ty2.fill_(0.5)
       tx3.fill_(0.5)
       ty3.fill_(0.5)
       tx4.fill_(0.5)
       ty4.fill_(0.5)
       tx5.fill_(0.5)
       ty5.fill_(0.5)

       coord_mask.fill_(1)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(50):
            if target[b][t*15+1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000
            gx0 = target[b][t*15+1] * nW
            gy0 = target[b][t*15+2] * nH
            gi0 = int(gx0)
            gj0 = int(gy0)
            gx1 = target[b][t*15+3] * nW
            gy1 = target[b][t*15+4] * nH
            gx2 = target[b][t*15+5] * nW
            gy2 = target[b][t*15+6] * nH
            gx3 = target[b][t*15+7] * nW
            gy3 = target[b][t*15+8] * nH
            gx4 = target[b][t*15+9] * nW
            gy4 = target[b][t*15+10] * nH
            gx5 = target[b][t*15+11] * nW
            gy5 = target[b][t*15+12] * nH


            best_n = 0 # 1 anchor box
            gt_box = [gx0/nW,gy0/nH,gx1/nW,gy1/nH,gx2/nW,gy2/nH,gx3/nW,gy3/nH,gx4/nW,gy4/nH,gx5/nW,gy5/nH]
            pred_box = pred_corners[b*nAnchors+best_n*nPixels+gj0*nW+gi0]
            conf = corner_confidence9(gt_box, pred_box) 
            coord_mask[b][best_n][gj0][gi0] = 1
            cls_mask[b][best_n][gj0][gi0]   = 1
            conf_mask[b][best_n][gj0][gi0]  = object_scale
            tx0[b][best_n][gj0][gi0]        = target[b][t*15+1] * nW - gi0
            ty0[b][best_n][gj0][gi0]        = target[b][t*15+2] * nH - gj0
            tx1[b][best_n][gj0][gi0]        = (target[b][t*15+3]-target[b][t*15+1]) * nW
            ty1[b][best_n][gj0][gi0]        = (target[b][t*15+4]-target[b][t*15+2]) * nH
            tx2[b][best_n][gj0][gi0]        = (target[b][t*15+5]-target[b][t*15+1]) * nW
            ty2[b][best_n][gj0][gi0]        = (target[b][t*15+6]-target[b][t*15+2]) * nH
            tx3[b][best_n][gj0][gi0]        = (target[b][t*15+7]-target[b][t*15+1]) * nW
            ty3[b][best_n][gj0][gi0]        = (target[b][t*15+8]-target[b][t*15+2]) * nH
            tx4[b][best_n][gj0][gi0]        = (target[b][t*15+9]-target[b][t*15+1]) * nW
            ty4[b][best_n][gj0][gi0]        = (target[b][t*15+10]-target[b][t*15+2]) * nH
            tx5[b][best_n][gj0][gi0]        = (target[b][t*15+11]-target[b][t*15+1]) * nW
            ty5[b][best_n][gj0][gi0]        = (target[b][t*15+12]-target[b][t*15+2]) * nH

            tconf[b][best_n][gj0][gi0]      = conf
            tcls[b][best_n][gj0][gi0]       = target[b][t*15]

            if conf > 0.5:
                nCorrect = nCorrect + 1

    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx0, tx1, tx2, tx3, tx4, tx5, ty0, ty1, ty2, ty3, ty4, ty5, tconf, tcls
           
class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)//num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.4
        self.seen = 0

    def forward(self, output, target):
        # Parameters
        t0 = time.time()
        nB = output.data.size(0) # deep scale on each grid
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2) # number of grids in height
        nW = output.data.size(3)

        # Activation
        output = output.view(nB, nA, (13+nC), nH, nW)
        x0     = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y0     = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        x1     = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        y1     = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        x2     = output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW)
        y2     = output.index_select(2, Variable(torch.cuda.LongTensor([5]))).view(nB, nA, nH, nW)
        x3     = output.index_select(2, Variable(torch.cuda.LongTensor([6]))).view(nB, nA, nH, nW)
        y3     = output.index_select(2, Variable(torch.cuda.LongTensor([7]))).view(nB, nA, nH, nW)
        x4     = output.index_select(2, Variable(torch.cuda.LongTensor([8]))).view(nB, nA, nH, nW)
        y4     = output.index_select(2, Variable(torch.cuda.LongTensor([9]))).view(nB, nA, nH, nW)
        x5     = output.index_select(2, Variable(torch.cuda.LongTensor([10]))).view(nB, nA, nH, nW)
        y5     = output.index_select(2, Variable(torch.cuda.LongTensor([11]))).view(nB, nA, nH, nW)
        
        conf   = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([12]))).view(nB, nA, nH, nW))
        cls    = output.index_select(2, Variable(torch.linspace(13,13+nC-1,nC).long().cuda()))
        cls    = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        t1     = time.time()

        # Create pred boxes
        pred_corners = torch.cuda.FloatTensor(12, nB*nA*nH*nW)
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        pred_corners[0]  = (x0.data.view_as(grid_x) + grid_x) / nW
        pred_corners[1]  = (y0.data.view_as(grid_y) + grid_y) / nH
        pred_corners[2]  = (x1.data.view_as(grid_x)) / nW + pred_corners[0]
        pred_corners[3]  = (y1.data.view_as(grid_y)) / nH + pred_corners[1]
        pred_corners[4]  = (x2.data.view_as(grid_x)) / nW + pred_corners[0]
        pred_corners[5]  = (y2.data.view_as(grid_y)) / nH + pred_corners[1]
        pred_corners[6]  = (x3.data.view_as(grid_x)) / nW + pred_corners[0]
        pred_corners[7]  = (y3.data.view_as(grid_y)) / nH + pred_corners[1]
        pred_corners[8]  = (x4.data.view_as(grid_x)) / nW + pred_corners[0]
        pred_corners[9]  = (y4.data.view_as(grid_y)) / nH + pred_corners[1]
        pred_corners[10] = (x5.data.view_as(grid_x)) / nW + pred_corners[0]
        pred_corners[11] = (y5.data.view_as(grid_y)) / nH + pred_corners[1]

        gpu_matrix = pred_corners.transpose(0,1).contiguous().view(-1,12)
        pred_corners = convert2cpu(gpu_matrix)
        t2 = time.time()


           
        # Build targets
        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx0, tx1, tx2, tx3, tx4, tx5, ty0, ty1, ty2, ty3, ty4, ty5, tconf, tcls = \
                       build_targets(pred_corners, target.data, self.anchors, nA, nC, nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)
        cls_mask   = (cls_mask == 1)
        nProposals = int((conf > 0.25).sum().item())
        tx0        = Variable(tx0.cuda())
        ty0        = Variable(ty0.cuda())
        tx1        = Variable(tx1.cuda())
        ty1        = Variable(ty1.cuda())
        tx2        = Variable(tx2.cuda())
        ty2        = Variable(ty2.cuda())
        tx3        = Variable(tx3.cuda())
        ty3        = Variable(ty3.cuda())
        tx4        = Variable(tx4.cuda())
        ty4        = Variable(ty4.cuda())
        tx5        = Variable(tx5.cuda())
        ty5        = Variable(ty5.cuda())

        tconf      = Variable(tconf.cuda())
        tcls       = Variable(tcls[cls_mask].long().cuda())
        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())
        cls        = cls[cls_mask].view(-1, nC)  
        t3 = time.time()

        # Create loss
        # Create loss
        loss_x0    = self.coord_scale * nn.SmoothL1Loss(size_average=False)(x0*coord_mask, tx0*coord_mask)/2.0
        loss_y0    = self.coord_scale * nn.SmoothL1Loss(size_average=False)(y0*coord_mask, ty0*coord_mask)/2.0
        loss_x1    = self.coord_scale * nn.SmoothL1Loss(size_average=False)(x1*coord_mask, tx1*coord_mask)/2.0
        loss_y1    = self.coord_scale * nn.SmoothL1Loss(size_average=False)(y1*coord_mask, ty1*coord_mask)/2.0
        loss_x2    = self.coord_scale * nn.SmoothL1Loss(size_average=False)(x2*coord_mask, tx2*coord_mask)/2.0
        loss_y2    = self.coord_scale * nn.SmoothL1Loss(size_average=False)(y2*coord_mask, ty2*coord_mask)/2.0
        loss_x3    = self.coord_scale * nn.SmoothL1Loss(size_average=False)(x3*coord_mask, tx3*coord_mask)/2.0
        loss_y3    = self.coord_scale * nn.SmoothL1Loss(size_average=False)(y3*coord_mask, ty3*coord_mask)/2.0
        loss_x4    = self.coord_scale * nn.SmoothL1Loss(size_average=False)(x4*coord_mask, tx4*coord_mask)/2.0
        loss_y4    = self.coord_scale * nn.SmoothL1Loss(size_average=False)(y4*coord_mask, ty4*coord_mask)/2.0
        loss_x5    = self.coord_scale * nn.SmoothL1Loss(size_average=False)(x5*coord_mask, tx5*coord_mask)/2.0
        loss_y5    = self.coord_scale * nn.SmoothL1Loss(size_average=False)(y5*coord_mask, ty5*coord_mask)/2.0

        loss_conf  = nn.SmoothL1Loss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0
        #loss_cls   = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)
        loss_cls = 0
        loss_x     = loss_x0 + loss_x1 + loss_x2 + loss_x3 + loss_x4 + loss_x5 
        loss_y     = loss_y0 + loss_y1 + loss_y2 + loss_y3 + loss_y4 + loss_y5 
        if False:
            loss   = loss_x + loss_y + loss_conf + loss_cls
        else:
            loss   = loss_x + loss_y + 20*loss_conf
        t4 = time.time()

        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_corners : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))

        if False:
            print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, conf %f, cls %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.item(), loss_y.item(), loss_conf.item(), loss_cls.item(), loss.item()))
        else:
            print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, conf %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.item(), loss_y.item(), loss_conf.item(), loss.item()))
        
        return loss
