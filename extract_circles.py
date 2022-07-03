


import numpy as np
import matplotlib.cm as cm
import pandas as pd
import matplotlib.pyplot as plt
from tracking_class import Tracking
import matplotlib.ticker as tck
import matplotlib.pyplot as plt
from tracking_class import Tracking

# this script detects and records the position/radius of circles in video format data

root = '/Users/Jonas/OneDrive - UvA/PhD'
path = '/video tracking'
name  ='/Walking_slope_2'
ext = '.mp4'
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
x_max = 640
y_min = 0
y_max =480

ROI = window(x_min,x_max,y_min,y_max)

# alternatively, indicate coordinates for each particle:
# ROI = coords([[800,900],[700,700]])
params = {'p1':100,
          'p2': 19, #tune this parameter for detection threshold
          'Nframe':0, #amount of frames to analyze, 0 = all 
          'check':True,'overwrite':False,'save':True, 
          'r_ROI':20, # ROI window width/height (pixels)
          'filepath':filepath, 
          'r_obj':[8,11], #expected circle radius 
          'ROI':ROI,
          'window':[x_min,x_max,y_min,y_max]}

f = Tracking(filepath = filepath)
f.set_parameters(params)
f.start_tracking()


