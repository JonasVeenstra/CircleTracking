import numpy as np
import matplotlib.pyplot as plt
from tracking_classv2 import Tracking
#
# this script detects and records the position/radius of circles in video format data
root = '/Users/Jonas_1/Documents/LocalData/microcilia/data/CarpetCiliaLinseed'
path = '/'
name  ='1800'
ext = '.avi'

filepath = [root,path,name,ext]
params = {'p1':50,
          'p2': 14, #tune this parameter for detection threshold
          'MinDist':20, #minimum distance between particles (only on frame 0)
          #'Nframe':100, #amount of frames towindow()
          'check':True,'overwrite':True,
          'r_ROI':16, # ROI window width/height (pixels)
          'r_obj':[4,7], #expected circle radius 
          'filepath':filepath, 
          't0':0,'tf':1000, 

          'xr':[0,2000],'yr':[0,7000],
          'tif':True} #window

f = Tracking(filepath = filepath)
f.set_parameters(params)
f.start_tracking()
