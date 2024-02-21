import numpy as np
import matplotlib.pyplot as plt
from tracking_class import Tracking

# this script detects and records the position/radius of circles in video format data

root = '/Users/Jonas/Library/CloudStorage/OneDrive-UvA/data/limit cycles turn active matter/7hex'
path = '/coarsening'
name  ='/unsynchronized_7hex'
ext = '.mov'

filepath = [root,path,name,ext]


params = {'p1':100,
          'p2': 19, #tune this parameter for detection threshold
          'MinDist':20, #minimum distance between particles (only on frame 0)
        #   'Nframe':0, #amount of frames towindow()
          'check':True,'overwrite':False,
          'r_ROI':20, # ROI window width/height (pixels)
          'filepath':filepath, 
          'r_obj':[3,7], #expected circle radius 
          't0':0,'tf':12000,
          'xr':[100,1000],'yr':[500,1600]} #window

f = Tracking(filepath = filepath)
f.set_parameters(params)
f.start_tracking()
