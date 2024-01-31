import numpy as np
from tracking_class import Tracking
import matplotlib.pyplot as plt
from tracking_class import Tracking

# this script detects and records the position/radius of circles in video format data

# root = '/Users/Jonas'
# path = '/Library/CloudStorage/OneDrive-UvA/data/Topological defects/experiment/Melanie_N=3/movies'
# path = '/Downloads'
# name  ='/tm_obstacle_low'
# ext = '.mov'
# 
# root = '/Users/Jonas/Library/CloudStorage/OneDrive-UvA/data/limit cycles turn active matter/fatboy6'
# path = '/cycles v2'
# name  ='/passive'
# ext = '.mov'

root = '/Users/Jonas/Documents/LocalData'
path = '/dec2023'
name  ='/test3'
ext = '.avi'



filepath = [root,path,name,ext]



def window(x0,x1,y0,y1):
    # ROI window with x,y coords of rectangle in pixels.
    ROI = {'window': np.asarray([x0,x1,y0,y1])}
    return ROI
    
def coords(l):
    # x,y coords of individual windows 
    ROI = {'coords': np.asarray(l)}
    return ROI

# indicate ROI window (pixels):
x_min = 0
x_max = 9000
y_min = 0
y_max = 10100

ROI = window(y_min,y_max,x_min,x_max)
# alternatively, indicate coordinates for each particle:
# ROI = coords(np.asarray([[400,880],[760,900]])) 

#or interactively
ROI = {'interactive':True}
params = {'p1':100,
          'p2': 24, #tune this parameter for detection threshold
          'Nframe':0, #amount of frames towindow()
          'check':True,'overwrite':False,'save':True, 
          'r_ROI':50, # ROI window width/height (pixels)
          'filepath':filepath, 
          'r_obj':[15,20], #expected circle radius 
          'ROI':ROI,
          'window':[x_min,x_max,y_min,y_max],
          't0':0,'tf':1200}

f = Tracking(filepath = filepath)
f.set_parameters(params)
f.start_tracking()

