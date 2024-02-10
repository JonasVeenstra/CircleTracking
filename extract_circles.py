import numpy as np
import matplotlib.pyplot as plt
from tracking_class import Tracking

# this script detects and records the position/radius of circles in video format data

root = '/Users/Jonas/Documents/LocalData'
path = '/feb2024'
name  ='/eps=-0.01'
ext = '.mov'

filepath = [root,path,name,ext]


params = {'p1':100,
          'p2': 21, #tune this parameter for detection threshold
          'MinDist':20, #minimum distance between particles (only on frame 0)
        #   'Nframe':0, #amount of frames towindow()
          'check':True,'overwrite':False,'save':True, 
          'r_ROI':25, # ROI window width/height (pixels)
          'filepath':filepath, 
          'r_obj':[5,8], #expected circle radius 
          't0':0,'tf':12000}

f = Tracking(filepath = filepath)
f.set_parameters(params)
f.start_tracking()

