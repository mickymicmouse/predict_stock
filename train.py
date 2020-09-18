# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:51:21 2020

@author: seungjun
"""

import matplotlib.pyplot as plt
import FinanceDataReader as fdr
import numpy as np
import os 
import random

import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score 


### torch
import torch.optim as optim
import torch
import torchvision
import torch.backends.cudnn as cudnn
import shutil

"""
mixed_precision=True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed
"""

##hand made
import pre_data
import print_Candle
import dataloader
import models
from utils import *

global best_acc
best_acc=1000

def save_checkpoint(state, is_best, checkpoint, filename):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
 

def train(figures, labels, model, opt, Stock_name, use_cuda):
    opt=opt
    best_acc=0
    model = model
    optimize = optim.Adam(model.parameters(), lr = opt.lr, betas = (0.5, 0.999))
    criterion = torch.nn.BCELoss(size_average=True).cuda()
    title = 'CANDLE'
    
    if opt.resume:
        print('Resuming from checkpoint')
        assert os.path.isfile(opt.resume),'Error: no checkpoint dir'
        opt.checkpoint = os.path.dirname(opt.resume)
        
        checkpoint = torch.load(opt.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        logger = Logger(os.path.join(opt.checkpoint, 'log.txt'), title = title,resume = True)
    else:
        os.mkdir(opt.checkpoint+'_'+Stock_name)
        
        logger = Logger(os.path.join(opt.checkpoint+'_'+Stock_name, 'log.txt'), title= title)
        logger.set_names(['Train_loss','Valid_loss','Valid_acc'])
    
    for epoch in tqdm(range(opt.start_epoch, opt.finish_epoch)):
        
        batch = opt.batch
        train_ratio = opt.train_ratio
        train_num = int(train_ratio*len(labels))
    
        train_gen = dataloader.single_stock_generator(figures[:train_num], labels[:train_num], batch, dimension)
        

        
        
        for batch_idx, (inputs, targets) in enumerate(train_gen):
            model.train()
            losses = AVERAGEMETER()
            
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            
            optimize.zero_grad()
            
            train_result = model(inputs)
            
            
            loss = criterion(train_result, targets)
            loss.backward(retain_graph=True)
            optimize.step()
            #print(batch_idx)
            losses.update(loss.item(), inputs.size(0))
        
        test_gen = dataloader.single_stock_generator(figures[train_num:], labels[train_num:], batch, dimension)
    
        for batch_idx2, (inputs2, targets2) in enumerate(test_gen):
            model.eval()
            vlosses=AVERAGEMETER()
            vacces = AVERAGEMETER()
            if use_cuda:
                inputs2, targets2 = inputs2.cuda(), targets2.cuda()
            inputs2, targets2 = torch.autograd.Variable(inputs2), torch.autograd.Variable(targets2)
            
            with torch.no_grad():
                inputs2, targets2 = torch.autograd.Variable(inputs2), torch.autograd.Variable(targets2)    
                valid_result = model(inputs2)
            vloss = criterion(valid_result, targets2)
            valid_result=valid_result.cpu().detach().numpy()
            for y in range(len(valid_result)):
                if valid_result[y]>=0.5:
                    valid_result[y]=1
                else:
                    valid_result[y]=0

            vacc = accuracy_score(valid_result, targets2.cpu().detach().numpy())
            
            vacces.update(vacc,inputs2.size(0))
            vlosses.update(vloss.item(),inputs2.size(0))
        
        train_loss= losses.avg
        valid_loss= vlosses.avg
        valid_acc=vacces.avg
        
        is_best = valid_loss<best_acc
        best_acc = min(valid_loss, best_acc)
        
        logger.append([train_loss, valid_loss ,valid_acc])
        if is_best==True:

            save_checkpoint({
                    'epoch':epoch+1,
                    'state_dict':model.state_dict(),
                    'loss':valid_loss,
                    'best_acc':best_acc
                    }, is_best, checkpoint=opt.checkpoint+'_'+Stock_name, filename = 'model.pth.tar')
    logger.close()
    logger.plot()
    savefig(os.path.join(opt.checkpoint+'_'+Stock_name,'log.png'))
    print('\nEpoch: [%d | %d] best_acc: %f' % (epoch + 1, opt.finish_epoch, best_acc))
    print('Finish Best is: '+ str(best_acc))
        
    

def main(Stock_name, Stock_price, figures, labels):
    Stock_name = Stock_name
    os.environ['CUDA_VISIBLE_DEVICES']=opt.device
    use_cuda = torch.cuda.is_available()
    batch = opt.batch
    if opt.manualSeed is None:
        opt.manualSeed=random.randint(1,10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    
    cudnn.benchmark = True
    """
    tmp_data = next(train_gen)
    print("Chart image shape : ",np.shape(tmp_data[0]))
    print("Label shape :",np.shape(tmp_data[1]))
    """
    

    model_name = opt.model_name
    if model_name =='CNN':
        model = models.CNN_model().cuda()
        
        model.apply(models.weights_init)
        print('model1 complete')
        train(figures, labels, model, opt, Stock_name, use_cuda)
        
    elif model_name =='CNN2':
        model = models.CNN_model()
    
    
    
    
    
    
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_epoch', type = int, default=0)
    parser.add_argument('--finish_epoch', type = int, default=10)
    parser.add_argument('--batch', type = int, default=16)
    
    parser.add_argument('--device', type = str, default='0')
    
    parser.add_argument('--manualSeed', type = int, default=None)
    parser.add_argument('--model_name', type = str, default='CNN')
    
    #seq_len, dimension, start_date,manualSeed,select, train_ratio, resume=str, model=str
    
    parser.add_argument('--start_date', type = str, default='2020-01-01')
    parser.add_argument('--train_ratio', type = float, default=0.8)
    parser.add_argument('--resume', type = str)
    parser.add_argument('--model', type = str, default='CNN')
    parser.add_argument('--select', type = str, default='low_up.0')
    parser.add_argument('--lr', type = float, default=0.001)
    parser.add_argument('--checkpoint', type = str, default='checkpoint')
    

    opt=parser.parse_args()
    
    """
    device = torch_util.select_device(opt.device,apex=mixed_precision, batch_size=opt.batch)
    if device.type == 'cpu':
        mixed_precision=False
        
    """
    print('start train')
    selected = opt.select
    df_result = pre_data.naver(select = 'quant.0')

    dimension = 48
    seq_len = 20
    period = 20
    pb=2
    seq_len=20
    start_date = opt.start_date

    top_list, top = pre_data.bollinger(period=period,pb=2,pre=1,min_per=1,start_date = start_date,df_result=df_result)

    Sale_com, Buy_com = pre_data.second_check(top_list, top)
    print(Buy_com)
    
    for i in range(len(Buy_com)):   
        
        selected_com = Buy_com[i]
        selected_code = top['Symbol'][top['Name']==selected_com]
        #stock_price = fdr.DataReader(selected_code.values[0], start_date)
        stock_price = fdr.DataReader('036570', start_date)
        figures, labels = print_Candle.ohlc2cs(df=stock_price, dimension=dimension, seq_len=seq_len)
        figures = figures/255.0
        print(np.shape(labels), np.shape(figures))
        
        main(selected_com, stock_price ,figures, labels)
        
    
