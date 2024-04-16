import numpy as np
import matplotlib.pyplot as plt
from tracking_class import Tracking

# this script detects and records the position/radius of circles in video format data
root = '/Users/Jonas/Library/CloudStorage/OneDrive-UvA/data/limit cycles turn active matter/grain substrate'
path = '/odd worm on grain'
name  ='/DSC_1560'
ext = '.mov'

filepath = [root,path,name,ext]

params = {'p1':100,
          'p2': 26, #tune this parameter for detection threshold
          'MinDist':20, #minimum distance between particles (only on frame 0)
        #   'Nframe':0, #amount of frames towindow()
          'check':True,'overwrite':False,
          'r_ROI':30, # ROI window width/height (pixels)
          'r_obj':[5,9], #expected circle radius 
          'filepath':filepath, 
          't0':0,'tf':16000,
          'xr':[0,1080],'yr':[0,1920]} #window

f = Tracking(filepath = filepath)
f.set_parameters(params)
f.start_tracking()
