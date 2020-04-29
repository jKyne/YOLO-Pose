import os
import time
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import scipy.io
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn

from darknet import Darknet
from PoseNet import DistanceNet, RotationNet
import dataset
from utils import *
from MeshPly import MeshPly

# Create new directory
def makedirs(path):
    if not os.path.exists( path ):
        os.makedirs( path )

def valid(datacfg, cfgfile, weightfile, weightfile_distance, weightfile_rotation, outfile):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    # Parse configuration files
    options      = read_data_cfg(datacfg)
    valid_images = options['valid']
    #meshname     = options['mesh']
    backupdir    = options['backup']
    name         = options['name']
    if not os.path.exists(backupdir):
        makedirs(backupdir)

    # Parameters
    prefix       = 'results'
    seed         = int(time.time())
    #gpus         = '0,1,2,3'     # Specify which gpus to use
    gpus         = options['gpus']
    test_width   = 416
    test_height  = 416
    torch.manual_seed(seed)
    use_cuda = True
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)
    save            = False
    testtime        = True
    use_cuda        = True
    num_classes     = 1
    testing_samples = 0.0
    eps             = 1e-5
    notpredicted    = 0 
    conf_thresh     = 0.1
    nms_thresh      = 0.4
    match_thresh    = 0.5
    if save:
        makedirs(backupdir + '/test')
        makedirs(backupdir + '/test/gt')
        makedirs(backupdir + '/test/pr')

    # To save
    testing_error_trans = 0.0
    testing_error_angle = 0.0
    testing_error_pixel = 0.0
    errs_2d             = []
    errs_3d             = []
    errs_trans          = []
    errs_angle          = []
    errs_corner2D       = []
    preds_trans         = []
    preds_rot           = []
    preds_corners2D     = []
    gts_trans           = []
    gts_rot             = []
    gts_corners2D       = []


    # Get validation file names
    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]
    
    # Specicy model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model = Darknet(cfgfile)
    model_2     = DistanceNet()
    model_3     = RotationNet()

    #model.print_network()
    model.load_weights(weightfile)    
    model_2.load_state_dict(torch.load(weightfile_distance))
    model_3.load_state_dict(torch.load(weightfile_rotation))
    
    model.cuda()
    model.eval()
    model_2.cuda()
    model_3.cuda()


    # Get the parser for the test dataset
    valid_dataset = dataset.listDataset(valid_images, shape=(test_width, test_height),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),]))
    valid_batchsize = 1

    # Specify the number of workers for multiple processing, get the dataloader for the test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batchsize, shuffle=False, **kwargs) 

    logging("   Testing {}...".format(name))
    logging("   Number of test samples: %d" % len(test_loader.dataset))
    # Iterate through test batches (Batch size for test data is 1)
    count = 0
    z = np.zeros((3, 1))
    out_file = open('data/valid_results.txt', 'w')
    for batch_idx, (data, target, gt_T, gt_R) in enumerate(test_loader):
        
        t1 = time.time()
        # Pass data to GPU
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        
        # Wrap tensors in Variable class, set volatile=True for inference mode and to use minimal memory during inference
        data = Variable(data, volatile=True)
        
        # Forward pass
        output = model(data).data  
        # Using confidence threshold, eliminate low-confidence predictions
        all_boxes_1 = get_region_boxes(output.data, conf_thresh, num_classes) 
        all_boxes = np.reshape(np.array(all_boxes_1),(-1,15))
        all_boxes = torch.from_numpy(all_boxes).float()        
        if use_cuda:
            all_boxes = all_boxes.cuda()
        output_T = model_2(all_boxes).float()
        if use_cuda:
            gt_T = gt_T.cuda()
            gt_R = gt_R.cuda()
        input_final = torch.cat((output_T, all_boxes),1)
        output_R = model_3(input_final)                      
        control_points = torch.FloatTensor([[[0.1,0,0]],[[0,0.1,0]],[[-0.1,0,0]],[[0,-0.1,0]]]).cuda()
        gt_proj = torch.zeros(output.size(0), 4, 3).cuda()
        pred_proj = torch.zeros(output.size(0), 4, 3).cuda()
        # Iterate through all images in the batch
        for j in range(output.size(0)):
        
            # For each image, get all the predictions
            boxes   = all_boxes_1[j]
            plot_boxes_polygon(data[j], boxes, "","",savename='data/Results/batch%s_%s.jpeg'%(str(batch_idx),str(j)), class_names=['QR_Code'])
            if boxes ==[[]]:
                continue
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
                gt_proj[j][i] = gt_T[j] + torch.mm(gt_rotation, control_points[i].t()).view(-1)
                pred_proj[j][i] = output_T[j] + torch.mm(pred_rotation,control_points[i].t()).view(-1)
            
            loss = 0
            loss_tmp = torch.zeros(4,4).cuda()
            loss_m = torch.zeros(4).cuda()
            for m in range(4):
                for n in range(4):
                    loss_tmp[m][n] = nn.MSELoss(size_average=True)(pred_proj[j][m],gt_proj[j][n])
            loss_m, index = torch.min(loss_tmp,dim=1)
            loss = torch.mean(loss_m)
            out_file.write('batch%d_%d,%04f\n'%(batch_idx,j,loss))








if __name__ == '__main__':
    import sys
    if len(sys.argv) == 4:
        datacfg = sys.argv[1]
        cfgfile = sys.argv[2]
        weightfile = sys.argv[3]
        weightfile_distance = weightfile.replace('model','model_distance').replace('weights','pkl')
        weightfile_rotation = weightfile.replace('model','rotation').replace('weights','pkl')
        outfile = 'comp4_det_test_'
        valid(datacfg, cfgfile, weightfile, weightfile_distance, weightfile_rotation, outfile)

    else:
        print('Usage:')
        print(' python valid.py datacfg cfgfile weightfile')
