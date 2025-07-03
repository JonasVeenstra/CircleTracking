import numpy as np
import matplotlib.pyplot as plt
from tracking_class import Tracking
#
# this script detects and records the position/radius of circles in video format data
root = 'D:/damien'
path = '/VDP/'
name  ='2hexsym'
ext = '.avi'

filepath = [root,path,name,ext]
#THERE IS A LEG THAT FELL AND I SAW IT AFTER TAKING MEASURMENTS D:\damien\UnitCell\k250\UCk250pw15 (1).mp4
#small hole 264
#full lattive 265
#to track k90pw80 
params = {'p1':100,
          'p2': 18, #tune this parameter for detection threshold
          'MinDist':40, #minimum distance between particles (only on frame 0)
          #'Nframe':100, #amount of frames towindow()
          'check':True,'overwrite':False,
          'r_ROI':50, # ROI window width/height (pixels)
          'r_obj':[10,25], #expected circle radius 
          'filepath':filepath, 
          't0':0,'tf':1000,
          'xr':[0,4000],'yr':[0,4000],
          'tif':True} #window

f = Tracking(filepath = filepath)
f.set_parameters(params)
f.start_tracking()
