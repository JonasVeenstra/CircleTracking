import numpy as np
import matplotlib.pyplot as plt
from tracking_class import Tracking

# this script detects and records the position/radius of circles in video format data
# root = '/Users/Jonas/Library/CloudStorage/OneDrive-UvA/data/limit cycles turn active matter/grain substrate'
# path = '/odd worm on grain'
# name  ='/DSC_1560'z

root = "/Users/Jonas/Library/CloudStorage/OneDrive-UvA/data/Topological defects/experiment/trihex N=4/"
path = 'data/'
name = '/OUTlow'
ext = '.mp4'

filepath = [root,path,name,ext]

params = {'p1':100,
          'p2': 14, #tune this parameter for detection threshold
          'MinDist':45, #minimum distance between particles (only on frame 0)
        #   'Nframe':0, #amount of frames towindow()
          'check':True,'overwrite':False,
          'r_ROI':40, # ROI window width/height (pixels)
          'r_obj':[12,15], #expected circle radius 
          'filepath':filepath, 
          't0':0,'tf':29999,
          'xr':[0,3080],'yr':[0,5850],
          'affix':'','nskip':1} #window

f = Tracking(filepath = filepath)
f.set_parameters(params)
f.start_tracking()


