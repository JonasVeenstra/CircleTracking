import numpy as np
import matplotlib.pyplot as plt
from tracking_classv2 import Tracking
import os
#
# this script detects and records the position/radius of circles in video format data
root = '/Volumes/Home Bartololab3/SystematicCharacterization/Circle/Row6_h7-12/'
path = '/'
name  ='3100'
ext = '.avi'

filepath = [root,path,name,ext]
params = {'p1':30, #parameter that determines whether to look at edges or at bulk (100 is bulk I believe)
          'p2': 23, #detection threshold parameter
          'MinDist':700, #minimum distance between particles (only on frame 0)
          #'Nframe':100, #amount of frames towindow()
          'check':False,'manual':False,'overwrite':True,
          'r_ROI':35, # ROI window width/height (pixels)
          'r_obj':[20,30], #expected circle radius 
          'filepath':filepath, 
          't0':1,'tf':10000, 
          'xr':[0,500],'yr':[0,6000],
          'tif':False} #window


params['check'] = True
params['manual'] = True
f = Tracking(filepath = filepath) 
f.set_parameters(params)
f.start_tracking()



# for n in ['k2','k3','k4','k5','k6']:
#     root = '/Volumes/Home Bartololab3/CILIABridge/ElasticCilia/1D/'z
#     root += n
# 
# lst = os.listdir(root+path)
# 
# try:
#     lst.remove('pickle')
#     lst.remove('.DS_Store')
# except ValueError:
#     pass
# 
# lst = sorted(
#     lst,
#     key=lambda f: abs(int(f.replace('.avi', '')))
# )
# lst=lst[-6:]
# print(lst)
# 
# 
# 
# for l in lst: 
#     print(l)
#     filepath = [root,path,l.split('.')[0],'.'+l.split('.')[1]]
#     f = Tracking(filepath = filepath) 
#     f.set_parameters(params)
#     f.start_tracking()
