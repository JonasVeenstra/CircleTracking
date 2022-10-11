


import numpy as np
import matplotlib.cm as cm
import pandas as pd
import matplotlib.pyplot as plt
from tracking_class import Tracking
import matplotlib.ticker as tck
import matplotlib.pyplot as plt
from tracking_class import Tracking

# this script detects and records the position/radius of circles in video format data

root = '/Users/Jonas'
path = '/Downloads'
name  ='/DSC_0366_Trim'
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
x_max = 1980
y_min = 0
y_max = 1920

ROI = window(y_min,y_max,x_min,x_max)
# 
# t=100
# small = 0.5*t
# big = np.sqrt(3)/2 * t
# 
# lb = [170,230]
# clist=[]
# nx = np.arange(24)
# ny = np.arange(5)
# a = np.sort(np.concatenate((nx[1::4],nx[2::4])))
# b = np.sort(np.concatenate((nx[::4],nx[3::4])))
# 
# dx = np.empty((25,))
# dy = np.empty((25,))
# dx[::2] = t
# dx[1::2] = small
# dx[0]=0
# dcum = np.cumsum(dx)
# 
# dy[::4]=0
# dy[3::4]=0
# dy[1::4]=big
# dy[2::4]=big
# 
# for j in nx:    
#     for i in ny:
#         x = lb[0] + dcum[j] 
#         y = lb[1] - dy[j] 
# 
#         x+=-1.5*i -2.5*j
#         y+=1.2*j
#         c = [x,y + i* t* np.sqrt(3)]
# 
#         clist.append(c)


# ROI = coords(clist) #y,x


# alternatively, indicate coordinates for each particle:
# ROI = coords(np.asarray([[800,900],[700,700]]))
params = {'p1':100,
          'p2': 15, #tune this parameter for detection threshold
          'Nframe':0, #amount of frames towindow()
          'check':True,'overwrite':False,'save':True, 
          'r_ROI':55, # ROI window width/height (pixels)
          'filepath':filepath, 
          'r_obj':[7,10], #expected circle radius 
          'ROI':ROI,
          'window':[x_min,x_max,y_min,y_max]}



f = Tracking(filepath = filepath)
f.set_parameters(params)

f.start_tracking()


