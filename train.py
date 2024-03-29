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
from efficientnet_pytorch import EfficientNet

### torch
import torch.optim as optim
import torch
import torchvision as tv
import torch.backends.cudnn as cudnn
import shutil
from torch.utils.data import DataLoader

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
from dataloader import Pathdataset
import models
from utils import Logger, AVERAGEMETER, savefig

global best_acc
best_acc=0

def save_checkpoint(state, is_best, checkpoint, filename):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
 

def train(figures, labels, model, opt, Stock_name, use_cuda):
    opt=opt
    best_acc=0
    model = model
    optimize = optim.Adam(model.parameters(), lr = opt.lr, betas = (0.9, 0.999))
    criterion = torch.nn.CrossEntropyLoss()
    title = 'CANDLE'
    batch = opt.batch
    train_ratio = opt.train_ratio
    train_num = int(train_ratio*len(labels))
    print(figures[:train_num].shape)
    print(figures[train_num:].shape)
    
    means=[0,0,0,0]
    stds = [0,0,0,0]
    train = figures[:train_num]
    for i in range(4):
        means[i] = np.mean(train[:,:,:,i])
        stds[i]=np.std(train[:,:,:,i])

    print(means)
    print(stds)
    
    
    
    train_transform = tv.transforms.Compose([tv.transforms.ToPILImage(mode = 'RGBA'),
                            tv.transforms.Resize(dimension),
                            tv.transforms.ToTensor(),
                            tv.transforms.Normalize(mean = means, std = stds)
                            ])
    valid_transform = tv.transforms.Compose([tv.transforms.ToPILImage(mode = 'RGBA'),
                            tv.transforms.Resize(dimension),
                            tv.transforms.ToTensor(),
                            tv.transforms.Normalize(mean =means, std = stds)
                            ])
    
    trainset = Pathdataset(image = figures[:train_num], labels = labels[:train_num],test_mode = False, transform=train_transform)
    validset = Pathdataset(image = figures[train_num:], labels = labels[train_num:],test_mode = False, transform=valid_transform)
    
    trainloader = DataLoader(dataset=trainset, 
            batch_size=batch, shuffle=True, drop_last = True)
    validloader = DataLoader(dataset=validset, 
            batch_size=batch, shuffle=False, drop_last = True)
    
    if opt.resume:
        print('Resuming from checkpoint')
        assert os.path.isfile(opt.resume),'Error: no checkpoint dir'
        opt.checkpoint = os.path.dirname(opt.resume)
        
        checkpoint = torch.load(opt.resume)
        best_acc = checkpoint['best_acc']

        model.load_state_dict(checkpoint['state_dict'])
        logger = Logger(os.path.join(opt.checkpoint, 'log.txt'), title = title,resume = True)
    else:
        os.mkdir(opt.checkpoint+'_'+Stock_name)
        
        logger = Logger(os.path.join(opt.checkpoint+'_'+Stock_name, 'log.txt'), title= title)
        logger.set_names(['Train_loss','Valid_loss','Valid_acc'])
    
    for epoch in tqdm(range(opt.start_epoch, opt.finish_epoch)):
        
        losses = AVERAGEMETER()
        vlosses=AVERAGEMETER()
        vacces = AVERAGEMETER()

        
        
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):

            model.train()            
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            train_result = model(inputs)
            loss = criterion(train_result, targets)

            optimize.zero_grad()

            loss.backward(retain_graph = True)
            optimize.step()

            print(str(np.max(train_result.cpu().detach().numpy(),axis=1)))
            losses.update(loss.item(), inputs.size(0))
        
        for batch_idx, (inputs, targets) in enumerate(validloader):

            #model.eval()
            
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            valid_result = model(inputs)
            vloss = criterion(valid_result, targets)
            valid_result=valid_result.cpu().detach().numpy()
            #print(str(valid_result))
            valid_result = np.argmax(valid_result,axis=1)

            vacc = accuracy_score(valid_result, targets.cpu().detach().numpy())
            
            print('predicted '+str(np.count_nonzero(valid_result)))
            print('answer '+str(np.count_nonzero(targets.cpu().detach().numpy())))
            vacces.update(vacc,1)
            vlosses.update(vloss.item(),inputs.size(0))
        
        train_loss= losses.avg
        valid_loss= vlosses.avg
        valid_acc=vacces.avg
        
        is_best = valid_acc>best_acc
        best_acc = max(valid_acc, best_acc)
        
        logger.append([train_loss, valid_loss ,valid_acc])
        if is_best==True:

            save_checkpoint({
                    'epoch':epoch+1,
                    'state_dict':model.state_dict(),
                    'loss':valid_loss,
                    'best_acc':best_acc
                    }, is_best, checkpoint=opt.checkpoint+'_'+Stock_name, filename = 'model.pth.tar')
        print('\nEpoch: [%d | %d] train_loss: %f valid_loss: %f valid_acc: %f' % (epoch, opt.finish_epoch,train_loss,valid_loss, valid_acc))
    logger.close()
    logger.plot()
    savefig(os.path.join(opt.checkpoint+'_'+Stock_name,'log.png'))

    print('Finish Best is: '+ str(best_acc))
        
    

def main(Stock_name, Stock_price, figures, labels):
    Stock_name = Stock_name
    os.environ['CUDA_VISIBLE_DEVICES']=opt.device
    use_cuda = torch.cuda.is_available()
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
    elif model_name =='Eff':
        model=EfficientNet.from_pretrained('efficientnet-b3',num_classes=2).cuda()
        print('model efficientnet')
        train(figures, labels, model, opt, Stock_name, use_cuda)

    
    
    elif model_name =='FB':
        model = models.FBProphet()
    
    
    
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_epoch', type = int, default=0)
    parser.add_argument('--finish_epoch', type = int, default=100)
    parser.add_argument('--batch', type = int, default=64)
    
    parser.add_argument('--device', type = str, default='0')
    
    parser.add_argument('--manualSeed', type = int, default=None)
    parser.add_argument('--model_name', type = str, default='CNN')
    
    #seq_len, dimension, start_date,manualSeed,select, train_ratio, resume=str, model=str
    parser.add_argument('--add_v', type = int, default=0)    
    parser.add_argument('--start_date', type = str, default='2000-01-01')
    parser.add_argument('--train_ratio', type = float, default=0.8)
    parser.add_argument('--resume', type = str)

    parser.add_argument('--select', type = str, default='low_up.0')
    parser.add_argument('--lr', type = float, default=0.00005)
    parser.add_argument('--checkpoint', type = str, default='checkpoint')
    

    opt=parser.parse_args()
    
    """
    device = torch_util.select_device(opt.device,apex=mixed_precision, batch_size=opt.batch)
    if device.type == 'cpu':
        mixed_precision=False
        
    """
    print('start train')
    selected = opt.select
    try:
        df_result = pre_data.naver(select = selected)
    except:
        print('re-connect naver finance')
        df_result = pre_data.naver(select = selected)
    dimension = 50
    seq_len = 20
    period = 20
    pb=2
    start_date = opt.start_date

    top_list, top = pre_data.bollinger(period=period,pb=2,pre=1,min_per=1,start_date = start_date,df_result=df_result)

    Sale_com, Buy_com = pre_data.second_check(top_list, top)
    print(Buy_com)
    """
    for i in range(len(Buy_com)):   
        
        selected_com = Buy_com[i]
        selected_code = top['Symbol'][top['Name']==selected_com]
        #stock_price = fdr.DataReader(selected_code.values[0], start_date)
        stock_price = fdr.DataReader('010140', start_date)
        figures, labels = print_Candle.ohlc2cs(df=stock_price, dimension=dimension, seq_len=seq_len, add_v = opt.add_v)

        print(np.shape(labels), np.shape(figures))
        
        main(selected_com, stock_price ,figures, labels)
    """
    stock_li=['010140','010140','019170','003000']
    labels_final = []
    for i in range(4):   
        print(str(i))
        selected_com = "test"
        # selected_code = top['Symbol'][top['Name']==selected_com]
        #stock_price = fdr.DataReader(selected_code.values[0], start_date)
        stock_price = fdr.DataReader(stock_li[i], start_date)
        figures, labels = print_Candle.ohlc2cs(df=stock_price, dimension=dimension, seq_len=seq_len, add_v = opt.add_v)
        if i ==0:
            figures_final = figures
        else:

            figures_final = np.concatenate((figures_final, figures),axis=0)
        labels_final.extend(labels)

    print(np.shape(labels_final), np.shape(figures_final))
        
    main(selected_com, stock_price ,figures_final, labels_final)
        

        
    
