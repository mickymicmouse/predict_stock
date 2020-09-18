# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 23:14:38 2020

@author: seungjun
"""

import matplotlib.pyplot as plt
import matplotlib as mpl 
import matplotlib.font_manager as fm

import pandas as pd
import numpy as np
import os 
import random

from pathlib import Path
from shutil import copyfile, move


### torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torch



def single_stock_generator(chart, labels, batch, dimension):
    #output = [chart, labels]
    
    num = len(labels)//batch
    for j in range(num):
        stock_batch = np.zeros(shape = (batch, 3, dimension, dimension))
        label_batch = np.zeros(shape=(batch,))
        idx=j*batch
        for i in range(batch):
            
            #idx = np.random.randint(len(labels))
            kk = np.transpose(chart[idx],(2,0,1))
            stock_batch[i]=kk
            label_batch[i]=labels[idx]
            idx+=1
        stock_batch = torch.tensor(stock_batch).float()
        label_batch = torch.tensor(label_batch).float()
        yield stock_batch, label_batch

