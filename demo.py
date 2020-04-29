import os
import time
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import scipy.io
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from darknet import Darknet
from PoseNet import DistanceNet, RotationNet
import dataset
from utils import *
from MeshPly import MeshPly
import cv2
from PIL import Image

# Create new directory
def makedirs(path):
    if not os.path.exists( path ):
        os.makedirs( path )

def numpy2Tensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  


def valid(datacfg, cfgfile, weightfile, weightfile_distance, weightfile_rotation, outfile):

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

    '''
    # Get the parser for the test dataset
    valid_dataset = dataset.listDataset(valid_images, shape=(test_width, test_height),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),]))
    '''

    cap = cv2.VideoCapture('./data/file_2.mp4')
    while(cap.isOpened()):
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        for i in range(int(frames)):
            time_1 = time.time()
            ret, img = cap.read()
            img = cv2.resize(img,(416,416))
            data = numpy2Tensor(img)
            if use_cuda:
                data = data.cuda()
        
            # Wrap tensors in Variable class, set volatile=True for inference mode and to use minimal memory during inference
            data = Variable(data, volatile=True)           
            # Forward pass
            output = model(data).data  
            
            # Using confidence threshold, eliminate low-confidence predictions
            boxes_1 = get_region_boxes(output, conf_thresh, num_classes) 
            if boxes_1 == [[]]: 
                img = data[0].cpu()
                img = transforms.ToPILImage()(img).convert('RGB')
                img = img.resize((768,432),Image.ANTIALIAS)
                savename = 'data/Results/%07d.jpeg'%i
                print("save plot results to %s" % savename)
                img.save(savename)
                continue 
            boxes = np.reshape(np.array(boxes_1),(-1))
            boxes = torch.from_numpy(boxes).float().cuda()  
            output_distance = model_2(boxes).float().view(-1)
            input_final = torch.cat((output_distance, boxes),0)
            output_rotation = model_3(input_final)
            time_2 = time.time()
            distance_txt = "(" + ", ".join([str(round(a.item(),2)) for a in output_distance]) + ")"     
            rotation_txt = "(" + ", ".join([str(round(a.item(),1)) for a in output_rotation]) + ")" 
            plot_boxes_polygon(data[0], boxes_1[0],distance_txt, rotation_txt, savename='data/Results/%07d.jpeg'%i, class_names=['QR_Code'])
            if i%10 == 0:
                fps = 1.0/(time_2 - time_1)
                print("Average FPS:"+str(fps))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

           
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
