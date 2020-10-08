# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 16:11:45 2020

@author: seungjun
"""

import schedule
import time
import numpy as np
import datetime

def job():
    print('time is now')
schedule.every().thursday.at('16:34').do(job)

while True:
    schedule.run_pending()
    time.sleep(1)


