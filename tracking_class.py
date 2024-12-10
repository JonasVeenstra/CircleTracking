#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:25:09 2021

@author: Jonas
"""
from turtle import distance
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys,time
import pickle
import os
from scipy.spatial import distance_matrix

class data(object):
    'class for storing data'
    pass
class Tracking(object):
    def __init__(self,display=True,filepath=''):
        self.data = data()
        self.timestamps = []
        self.filepath=''.join(filepath)
        self.folderpath = ''.join(filepath[:2])
        self.filename = filepath[2] 
        self.picklepath = self.folderpath + '/pickle'
        self.input = True
        if display == True:
            print('File:        ' + (self.filepath))
        
    def set_parameters(self,params):
        self.filename += params['affix']
        self.params=params
        
    def start_tracking(self):
        self.f=self.params['t0'] #frame zero
        self.cap = cv2.VideoCapture(self.filepath)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.f)
        self.totalframes =  int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
        self.frames_to_track = np.min([self.params['tf'] - self.params['t0'],self.totalframes])

        if self.totalframes == -9223372036854775808:
            self.totalframes = 1

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if (self.cap.isOpened()== False):               
            print("Error opening video stream or file")

        print('TOTAL FRAMES: ',self.totalframes, f't0 = {self.params["t0"]}')
        self.read_frame()
        self.findcircles()
        self.N = len(self.circles)
        
        self.data.raw = [self.circles] #np.zeros((self.frames_to_track,self.N,3)) # t * N * 3 array with x,y,r data of detected circles
        # self.circles = np.asarray(self.circles)
        print('circle radius = ' + str(np.mean(self.circles[:,2])))
        # self.data.raw[0,:,:] = np.asarray(self.circles)
        self.f+=self.params['nskip']

        print('gogogo')
        self.run_tracker()

    def run_tracker(self):
        self.f=self.params['t0']+self.params['nskip']
        print(self.f,self.totalframes , self.f,self.params['tf']-self.params['t0'])
        while(self.f<self.totalframes and self.f<(self.params['tf'])):
            print(self.f)
            if self.ret == False:
                self.cap.release()
                print('end of tracking')
                break
            self.read_frame()
            self.findcircles()
            try:
                self.data.raw.append(self.circles)    
            except ValueError:
                print('particle number not conserved')
                break
            self.f+=self.params['nskip']
        self.savedata()

    def savedata(self):
        self.data.timestamps=self.timestamps
        self.extract_data()

    def read_frame(self):
        self.ret,frame = self.cap.read()
        self.timestamps.append(self.cap.get(cv2.CAP_PROP_POS_MSEC)/1e3)
        if self.ret==True:
            xr =  self.params['xr']
            yr =  self.params['yr']
            self.cimg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)[xr[0]:xr[1],yr[0]:yr[1]]
        elif self.f ==self.totalframes-1:
            pass
        else:
            print('warning, frame not found')


    def onclick(self,event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
        print (f'x = {ix}, y = {iy}')
        self.clicked_coordinate.append([ix,iy])
    def photocheck(self):
        
        
        while(self.params['check']):

            print(f'current N={len(self.circles)}')
            self.clicked_coordinate=[]
            fig,ax = self.photo_plot()
            cid = fig.canvas.mpl_connect('button_press_event', self.onclick)            
            
            # self.clicked_coordinate = np.unique(self.clicked_coordinate)
            # print(np.shape(self.clicked_coordinate))
            plt.show()

            self.clicked_coordinate = np.unique(self.clicked_coordinate,axis=0)

            if len(self.clicked_coordinate)==0: 
                print('len clicked coordinate =0')
                break

            for n,unit in enumerate(self.clicked_coordinate):   ####loop over all N ROI's
                self.unitnumber=n
                y = int(unit[0])            ##### CAREFUL: image input is of form [y,x] (cf. '1080x1920')
                x = int(unit[1])
                ROIr = self.params['r_ROI']
                self.xmin = np.max([0,np.max([x - ROIr])])
                self.xmax = np.min([np.max([x + ROIr,2*ROIr])])
                self.ymin = np.max([0,np.max([y - ROIr])])
                self.ymax = np.min([np.max([y + ROIr,2*ROIr])])        
                print(self.xmin,self.xmax,self.ymin,self.ymax)
                self.ROIimg = self.cimg[self.xmin:self.xmax,self.ymin:self.ymax]
                
                while True:
                    added_circle = self.circlefitter(self.ROIimg)
                    if len(added_circle)==0:
                        self.params['p2']+=-1
                    elif len(added_circle)==0:
                        self.params['p2']+=1
                    else: 
                        break
                
                added_circle[0][:2] +=  [self.ymin,self.xmin]#[y-int(ROIr),x-int(ROIr)]#HERE
                added_circle=np.array(added_circle[0])
                try:
                    self.circles=np.append(self.circles,[added_circle],axis=0)
                except ValueError:
                    # self.circles=np.append(self.circles,np.array([added_circle]))
                    self.circles = np.array([added_circle])



    def photo_plot(self):
        fig,ax = plt.subplots(1,figsize=(6,6))
        circles = np.asarray(self.circles)
        ax.imshow(cv2.cvtColor(self.cimg,cv2.COLOR_GRAY2BGR))        
        for n,i in enumerate(circles):
            x,y=[i[0],i[1]]
            ax.annotate(str(n),[x,y],c='white')
            ax.add_patch(plt.Circle((x,y),i[2],color='r',lw = 10,fill=False))
        return fig,ax
    def circlefitter(self,cimg):
        p1 = self.params['p1']
        p2 = self.params['p2']
        mindist = self.params['MinDist']
        r1,r2 = self.params['r_obj']
        try:
            circles = cv2.HoughCircles(cimg,3,1,minDist=mindist,param1=p1,param2=p2,minRadius=r1,maxRadius=r2)[:][:][0]
        except TypeError:
            circles = []
        except cv2.error as e:
            print('error')
            print(self.xmin,self.xmax,self.ymin,self.ymax)
            print(cimg)
            plt.imshow(cimg)
            plt.show()
            plt.imshow(self.ROIimg)
            plt.show()

            circles = self.circles[-1]

        return circles

    def manual_detection(self):
        self.clicked_coordinate=[]
        fig,ax = plt.subplots(1,figsize=(6,6))
        try:
            ax.imshow(cv2.cvtColor(self.ROIimg,cv2.COLOR_GRAY2BGR))        
        except cv2.error:
            print(self.ROIimg)
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick) 
        plt.show()       
        try:    
            self.clicked_coordinate=self.clicked_coordinate[0]
            self.clicked_coordinate.append(0)
            return np.array([self.clicked_coordinate])
        except IndexError:
            self.abort()
    def abort(self):
        self.savedata()
        print('tracking aborted but data saved')
        sys.exit()

    def findcircles(self):
        if self.f==self.params['t0']: #frame 0
            self.ROIimg = self.cimg
            self.circles = self.circlefitter(self.ROIimg)
            self.photocheck()
            time.sleep(0.1)
            plt.close()


        else: #rest of frames
            coords = self.circles
            circles = []
            for n,xy in enumerate(coords):   ####loop over all N ROI's
                self.unitnumber=n
                y = int(xy[0])            ##### CAREFUL: image input is of form [y,x] (cf. '1080x1920')
                x = int(xy[1])
                ROIr = self.params['r_ROI']
                self.xmin = np.max([0,np.max([x - ROIr])])
                self.xmax = np.min([np.max([x + ROIr,2*ROIr])])
                self.ymin = np.max([0,np.max([y - ROIr])])
                self.ymax = np.min([np.max([y + ROIr,2*ROIr])])
                self.ROIimg = self.cimg[self.xmin:self.xmax,self.ymin:self.ymax]
                run = True
                i=0
                while(run):
                    circle =self.circlefitter(self.ROIimg)
                    if i>3 or self.params['p2']<1:
                        print(self.circles[n]-np.array([self.xmin,self.ymin,0]))
                        print(circle)
                        circle = self.manual_detection()
                        


                    try :
                        if len(circle) == 0:    
                            i+=1
                            self.decrease_threshold()
                        elif len(circle)>1:
                            i+=1
                            self.increase_threshold()
                        elif len(circle)==1:
                            circle[0,:2] +=  [self.ymin,self.xmin]
                            circles.append(list(circle[0]))
                            run = False
                    except TypeError:
                        self.abort()

            self.circles = np.asarray(circles)

    def increase_threshold(self):
        self.params['p2'] = self.params['p2'] + 1
        print('p2 is now:    ',self.params['p2'])

    def decrease_threshold(self):
        self.params['p2'] = self.params['p2'] - 1
        print('p2 is now:    ',self.params['p2'])
    
    def extract_data(self):
        if not os.path.exists(self.picklepath):
            os.makedirs(self.picklepath)
        if os.path.isfile(self.picklepath) == False or self.overwrite == True:
            data = [np.array(self.data.raw),self.data.timestamps] #[[x,y,r],t]
            pickle.dump(data,open(self.picklepath  + self.filename,'wb'))
        else:
            print(f"The data has not been saved since overwrite = False and {self.filename} already exists")

    def load_data(self):
        with (open(self.picklepath + self.filename,'rb')) as openfile:
            while True:
                try:
                    print(self.picklepath + self.filename)
                    raw,timestamps = pickle.load(openfile)
                    break
                except EOFError:
                    print('no such file')
                    break
        return raw,timestamps