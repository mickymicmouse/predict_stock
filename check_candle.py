# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 18:50:44 2020

@author: seungjun
"""

import matplotlib.pyplot as plt
import pandas as pd

opens = [5000,4000,5000,6000,5000,6000,5000,5000,5000,4000]
closes = [6000,4000,5000,4000,6000,7600,5000,4500,4000,5000]
volume = [100,200,100,140,501,100,100,102,432,500]
vol = [1,2,1,2,1,2,3,4,2,1]

opens = pd.DataFrame(opens)
closes=pd.DataFrame(closes)
vol=pd.DataFrame(vol)
volume=pd.DataFrame(volume)



plt.bar(volume.index,height=volume[0], width=0.8)
plt.plot(vol[0])


import mpl_finance as mpl
from mpl_finance import volume_overlay


fig=plt.figure(figsize=(0.5,0.5),dpi=96)

ax1=fig.add_subplot(1,1,1)
plt.style.use('dark_background')
mpl.candlestick2_ochl(ax1, opens = opens[0],closes=closes[0],highs= opens[0],lows=closes[0],colorup='#77d879', colordown='#db3f3f',width=1)                      

ax1.grid(False)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)
ax1.axis('off')
fig.canvas.draw()

fig

ax2=fig.add_subplot(2,1,1)
mpl.volume_overlay(ax1, opens = opens[0],closes= closes[0], volumes = volume[0],colorup='#77d879', colordown='#db3f3f',width=1)
ax2.grid(False)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)
ax2.axis('off')
fig.canvas.draw()
fig.savefig('sample.png')
fig
import numpy as np
fig_np = np.array(fig.canvas.renderer._renderer)
figs = fig_np[:,:,:3]

import torchvision as tv


transform = tv.transforms.Compose([tv.transforms.ToPILImage(mode = 'RGB'),
                                   tv.transforms.Resize(48),
                                   tv.transforms.ToTensor(),
                                   #tv.transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225]),
                                   #tv.transforms.ToPILImage(mode='RGB')
])
sample = transform(figs)
sample=sample.numpy()
plt.imsave('sample.png',fig)


plt.close()

