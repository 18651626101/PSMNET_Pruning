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
from dataloader import KITTIloader2015 as ls
from dataloader import KITTILoader as DA
from xxx import Functions_quan as FQ
from models import *
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--datapath', default='/home/jump/dataset/kitti2015/training/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='./trained/pruned_0.6.tar',
                    help='load model')
parser.add_argument('--savemodel', default='./Ptrained_0.6/',
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

if args.datatype == '2015':
   from dataloader import KITTIloader2015 as ls
elif args.datatype == '2012':
   from dataloader import KITTIloader2012 as ls

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True), 
         batch_size= 8, shuffle= True, num_workers= 4, drop_last=False)

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

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

#print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

writer=SummaryWriter('log')

def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1

def BN_grad_zero():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            mask = (m.weight.data != 0)
            mask = mask.float().cuda()
            m.weight.grad.data.mul_(mask)
            m.bias.grad.data.mul_(mask)
        elif isinstance(m, nn.BatchNorm3d):
            mask3 = (m.weight.data != 0)
            mask3 = mask3.float().cuda()
            m.weight.grad.data.mul_(mask3)
            m.bias.grad.data.mul_(mask3)

def train(imgL,imgR,disp_L):
        model.train()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        disp_L = Variable(torch.FloatTensor(disp_L))

        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

        #---------
        mask = (disp_true > 0)
        mask.detach_()
        #----

        optimizer.zero_grad()
        
        if args.model == 'stackhourglass':
            output1, output2, output3 = model(imgL,imgR)
            output1 = torch.squeeze(output1,1)
            output2 = torch.squeeze(output2,1)
            output3 = torch.squeeze(output3,1)
            loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
        elif args.model == 'basic':
            output = model(imgL,imgR)
            output = torch.squeeze(output,1)
            loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

        loss.backward()
        if args.sr:
            updateBN()
        BN_grad_zero()
        optimizer.step()

        return loss.item()

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

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 20:
        lr = 0.15     #lr = 0.2 剪裁率0.8的时候学习率为0.2会发生梯度爆炸
    elif epoch <= 100:
        lr = 0.1      #lr = 0.1 剪裁率0.8的时候要改小
    elif epoch <= 150:
        lr = 0.02     #0.01
    elif epoch <= 200:
        lr = 0.01    #0.001
    else:
        lr = 0.005   #0.0001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def QFunction(value):
    qw = torch.tanh(value)
    qw = qw/torch.max(torch.abs(qw)).data.item()*0.5+0.5
    Qvalue = 2*FQ.QuantizeFunc.apply(qw)-1
    return Qvalue

def main():
	max_acc=0
	max_epo=0
	start_full_time = time.time()
	
	for epoch in range(1, args.epochs+1):
		total_train_loss = 0
		total_test_loss = 0
		adjust_learning_rate(optimizer,epoch)

		## training ##
		for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
			start_time = time.time()
			loss = train(imgL_crop,imgR_crop, disp_crop_L)
			print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
			total_train_loss += loss
			#tensorboardX可视化
		writer.add_scalar('Train/TotalTrainLoss',total_train_loss/len(TrainImgLoader),epoch)
		print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))
		
		## Test ##
		for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
			test_loss = test(imgL,imgR, disp_L)
			print('Iter %d 3-px error in val = %.3f' %(batch_idx, test_loss*100))
			total_test_loss += test_loss
		writer.add_scalar('Test/total 3-px error in val',total_test_loss/len(TestImgLoader)*100,epoch)
		print('epoch %d total 3-px error in val = %.3f' %(epoch, total_test_loss/len(TestImgLoader)*100))
		
		if (100-total_test_loss/len(TestImgLoader)*100) > max_acc:
			max_acc = 100-total_test_loss/len(TestImgLoader)*100
			max_epo = epoch
		print('MAX epoch %d total test accuracy = %.3f' %(max_epo, max_acc))

		
        

		#SAVE
        #test的时候保存正常的model，因为运行的时候会量化
        #到其他地方的时候保存量化的model
		if epoch%300 == 0:
			model_save = {k:QFunction(v) for k,v in model.state_dict().items() if "weight" in k or "bias" in k}
			model_save.update({k:v for k,v in model.state_dict().items() if "weight" not in k and "bias" not in k})
            
			savefilename = args.savemodel+'finetune_'+str(epoch)+'.tar'
			#Qsavefilename = args.savemodel+'Qfinetune_'+str(epoch)+'.tar'

			'''torch.save({
                'epoch': epoch,
                'state_dict': model_save,
                'train_loss': total_train_loss/len(TrainImgLoader),
                'test_loss': total_test_loss/len(TestImgLoader)*100,
			}, Qsavefilename)'''
            
			torch.save({
				'epoch': epoch,
				'state_dict': model.state_dict(),
				'train_loss': total_train_loss/len(TrainImgLoader),
				'test_loss': total_test_loss/len(TestImgLoader)*100,
			}, savefilename)
            
		
		print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))
	print(max_epo)
	print(max_acc)

writer.close()


if __name__ == '__main__':
   main()
