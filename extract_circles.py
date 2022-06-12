


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
name  ='/Basler_acA640-750um__23036100__20220610_183814017_1186'
ext = '.tiff'
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
ROI = window(0,600,0,450)

# alternatively, indicate coordinates for each particle:
# ROI = coords([[800,900],[700,700]])
params = {'p1':100,
          'p2': 19, #tune this parameter for detection threshold
          'Nframe':0, #amount of frames to analyze, 0 = all 
          'check':True,'overwrite':False,'save':True, 
          'r_ROI':34, # ROI window width/height (pixels)
          'filepath':filepath, 
          'r_obj':[6,12], #expected circle radius 
          'ROI':ROI}

f = Tracking(filepath = filepath)
f.set_parameters(params)
f.start_tracking()


