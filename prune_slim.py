from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from dataloader import KITTIloader2015test as ls
from dataloader import KITTILoader as DA
from dataloader import KITTILoader as DA

from models import *

from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='/home/jump/dataset/kitti2015/training/',
                    help='datapath') 
parser.add_argument('--loadmodel', default='./pretrained/pretrained_model_KITTI2015.tar',
                    help='load model')
parser.add_argument('--percent', type=float, default=0.75,
                    help='scale sparse rate (default: 0.5)')
# parser.add_argument('--loadmodel', default='./trained/Qfinetune_8bit.tar',
#                     help='load model')
parser.add_argument('--savemodel', default='./trained/',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

if not os.path.exists(args.savemodel):
    os.makedirs(args.savemodel)

if args.datatype == '2015':
   from dataloader import KITTIloader2015 as ls
elif args.datatype == '2012':
   from dataloader import KITTIloader2012 as ls

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(args.datapath)

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False), 
         batch_size= 8, shuffle= False, num_workers= 4, drop_last=False)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel:
    if os.path.isfile(args.loadmodel):
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'])

total = 0
total3 = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]
    elif isinstance(m,nn.BatchNorm3d):
        total3 += m.weight.data.shape[0]

bn = torch.zeros(total)
bn3 = torch.zeros(total3)
index = 0
index3 = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size
    elif isinstance(m, nn.BatchNorm3d):
        size3 = m.weight.data.shape[0]
        bn3[index3:(index3+size3)] = m.weight.data.abs().clone()
        index3 += size3

y, i = torch.sort(bn)
y3,i3 = torch.sort(bn3)
threshold_index = int(total * args.percent)
#print(threshold_index )
threshold = y[threshold_index].cuda()
threshold_index3 = int(total3 * args.percent)
threshold3 = y3[threshold_index3].cuda()
#pruned = 0
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(threshold).float().cuda()
        #pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.BatchNorm3d):
        weight_copy3 = m.weight.data.abs().clone()
        mask3 = weight_copy3.gt(threshold3).float().cuda()
        #pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask3)
        m.bias.data.mul_(mask3)
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask3.shape[0], int(torch.sum(mask3))))        

#pruned_ratio = pruned/total

'''torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))
print(newmodel)
model = newmodel'''

print('Pre-processing Successful!')

def test(imgL,imgR,disp_true):
        model.eval()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        with torch.no_grad():
            output3 = model(imgL,imgR)

        pred_disp = output3.data.cpu()

        #computing 3-px error#
        true_disp = disp_true
        index = np.argwhere(true_disp>0)
        disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
        correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
        torch.cuda.empty_cache()

        return 1-(float(torch.sum(correct))/float(len(index[0])))

def main():
    acc=0
    start_full_time = time.time()

    total_test_loss = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        test_loss = test(imgL,imgR, disp_L)
        print('Iter %d 3-px error in val = %.3f' %(batch_idx, test_loss*100))
        total_test_loss += test_loss

    acc = total_test_loss/len(TestImgLoader)*100
    print('total test error = %.3f' %(acc))	
    print('full finetune time = %.2f s' %(time.time() - start_full_time))
    
    savefilename = args.savemodel+'finetune_'+str(args.percent)+'.tar'

    torch.save({
        'state_dict': model.state_dict(),
        'test_loss': acc
    }, savefilename)

if __name__ == '__main__':
   main()