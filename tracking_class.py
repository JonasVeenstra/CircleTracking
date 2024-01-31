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
import sys
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
        self.params=params
        
    def start_tracking(self):
        self.f=self.params['t0'] #frame zero
        self.cap = cv2.VideoCapture(self.filepath)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.f)
        self.totalframes =  int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
        self.totalframes = self.totalframes - self.f
        if self.totalframes == -9223372036854775808:
            self.totalframes = 1
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if (self.cap.isOpened()== False):               
            print("Error opening video stream or file")
        if self.params['Nframe'] == 0:
            self.params['Nframe'] = self.totalframes
        print('total frames: ',self.params['Nframe'])
        
        self.params['Nframe'] = min(self.totalframes,self.params['Nframe'])
        self.read_frame()
        self.findcircles()
        self.N = len(self.circles)
        
        self.data.raw = np.zeros((self.params['Nframe'],self.N,3)) # t * N * 3 array with x,y,r data of detected circles
        self.circles = np.asarray(self.circles)
        
        print('circle radius = ' + str(np.mean(self.circles[:,2])))
        self.data.raw[0,:,:] = np.asarray(self.circles)
        self.f+=1
        
        self.run_tracker()

    def run_tracker(self):
        self.f=1
        while(self.f<self.params['Nframe'] and self.f<(self.params['tf']-self.params['t0'])):
            print(self.f)
            if self.ret == False:
                self.cap.release()
                break
            self.read_frame()
            self.params['ROI']= {'coords':self.data.raw[self.f-1,:,:2]}
            self.findcircles()
            
            try:
                self.data.raw[self.f,:,:] = np.asarray(self.circles)    
            except ValueError:
                # code something here to deal with nonconservative particle numbers
                print('particle number not conserved')
                break
            self.f+=1  
        
        if self.params['save'] == True:
            self.data.timestamps=self.timestamps
            self.extract_data()

    def read_frame(self):
        self.ret,frame = self.cap.read()
        self.timestamps.append(self.cap.get(cv2.CAP_PROP_POS_MSEC)/1e3)
        if self.ret==True:
            self.cimg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        elif self.f ==self.params['Nframe']-1:
            pass
        else:
            print('warning, frame not found')
    
    def findcircles(self):
        def circlefitter(cimg):
            p1 = self.params['p1']
            p2 = self.params['p2']
            r1,r2 = self.params['r_obj']
            self.circles = cv2.HoughCircles(cimg,3,1,20,param1=p1,param2=p2,minRadius=r1,maxRadius=r2)[:][:][0]
        
        self.coords =[]
        if 'interactive' in self.params['ROI']:
            self.circles=np.array([[]])
            fig,ax = self.photo_plot(a=1)
            def onclick(event):
                global ix, iy
                ix, iy = event.xdata, event.ydata
                print (f'x = {ix}, y = {iy}')
                self.coords.append([ix,iy])

            cid = fig.canvas.mpl_connect('button_press_event', onclick)            
            plt.show()
            
            self.params['ROI'] = {'coords': np.asarray(self.coords)}

        ROI = self.params['ROI']
        if ROI is None:
            circlefitter(self.cimg)
        elif 'window' in ROI:
            window = ROI['window']
            self.xmin = window[0]
            self.xmax = window[1]
            self.ymin = window[2]
            self.ymax = window[3]
            
            self.ROIimg = self.cimg[self.ymin:self.ymax,self.xmin:self.xmax]
            circlefitter(self.ROIimg)
            # plt.show()
            for ni,i in enumerate(self.circles):
                self.circles[ni][0] +=  self.xmin
                self.circles[ni][1] +=  self.ymin
            self.circles = self.circles[np.argsort(np.sqrt(self.circles[:, 0]**2+self.circles[:, 1]**2))]
            print('photocheck')
            self.photocheck(a='area')
            
        elif 'coords' in ROI:
            coords = ROI['coords']
            circles = []
            for n,unit in enumerate(coords):   ####loop over all N ROI's
                self.unitnumber=n
                y = int(unit[0])            ##### CAREFUL: image input is of form [y,x] (cf. '1080x1920')
                x = int(unit[1])
                ROIr = self.params['r_ROI']
                self.xmin = np.max([x - ROIr, self.params['window'][0]])
                self.xmax = np.min([np.max([x + ROIr,2*ROIr]), self.params['window'][3]])
                self.ymin = np.max([y - ROIr, self.params['window'][2]])
                self.ymax = np.min([np.max([y + ROIr,2*ROIr]), self.params['window'][1]])

        
                self.ROIimg = self.cimg[self.xmin:self.xmax,self.ymin:self.ymax]
                run = True
                p=0
                
                while(run):
                    try:
                        circlefitter(self.ROIimg)
                        circle = self.circles
                        if len(circle) > 1:
                            if self.params['p2'] == p:
                                print(f'cannot find particle {n+1}. skipping frame...1')
                                circles.append(self.data.raw[self.f - 1,n,:])
                                self.circles=np.asarray(circles)

                                fig,ax = self.photo_plot(a=0)
                                plt.show()
                                run = False
                            else:
                                print('Found more particles than expected: increasing threshold')
                                p = self.params['p2']
                                self.increase_threshold()
                        else:
                            circle[0][:2] +=  [self.ymin,self.xmin]#[y-int(ROIr),x-int(ROIr)]#HERE
                            circles.append(list(circle[0]))
                            run = False
                    except TypeError:
                        print('TYPEERROR')
                        self.decrease_threshold()
                        if self.params['p2'] == p or self.params['p2']==0:
                            print(f'cannot find particle {n+1}. skipping frame...2')
                            circles.append(self.data.raw[self.f - 2,n,:])
                            self.circles=np.asarray(circles)
                            fig,ax = self.photo_plot(a=0)
                            plt.show()
                        else:
                            print(f'particle {n+1} missing: decreasing threshold')
                            p = self.params['p2']
            self.circles = np.asarray(circles)
            self.photocheck(a='area')
            self.params['r_ROI']=15

    def increase_threshold(self):
        self.params['p2'] = self.params['p2'] + 1
        print('p2 is now:    ',self.params['p2'])

    def decrease_threshold(self):
        self.params['p2'] = self.params['p2'] - 1
        print('p2 is now:    ',self.params['p2'])

    def photo_plot(self,a=0,):
        fig,ax = plt.subplots(1,figsize=(12,6))
        circles = np.asarray(self.circles)
        if a==0:
            self.xmin,self.xmax,self.ymin,self.ymax = [int(np.min(circles[:,0])),int(np.max(circles[:,0])),int(np.min(circles[:,1])),int(np.max(circles[:,1]))]
        if 'window' in self.params['ROI']:
            rnge = self.params['ROI']['window']
            print(rnge)
            self.cimg = self.cimg[rnge[0]:rnge[1],rnge[2]:rnge[3]]
        print(1)
        for n,i in enumerate(circles):
            if 'window' in self.params['ROI']:
                x=i[0]-rnge[2]
                y=i[1]-rnge[0]
                if not 'interactive' in self.params['ROI']:
                    x,y=[i[0],i[1]]
                    ax.annotate(str(n),[x,y],c='white')
                    # ax.scatter(x,y,c='r',s=7)
                    ax.add_patch(plt.Circle((x,y),i[2],color='r',lw = 1,fill=False))
        
        ax.imshow(cv2.cvtColor(self.cimg,cv2.COLOR_GRAY2RGB))        
        return fig,ax

    def photocheck(self,a=0):
        while(self.params['check']):
            fig,ax= self.photo_plot(a=a)
            plt.show()
            
            N=len(self.circles)
            
            print(f'{N} circles detected. Increase (i) or decrease (d) threshold p2? otherwise ENTER. p2=',self.params['p2'])
            print('average radius: ', np.average(np.array(self.circles)[:,2]))
            yes = ''
            inc = 'i'
            dec = 'd'

            choice = input().lower()
            if choice == dec:
                self.decrease_threshold()
                self.findcircles()
            elif choice == inc:
                self.increase_threshold()
                self.findcircles()
            elif choice not in [yes,inc,dec]:
                sys.stdout.write("'m' or 'f'")
            elif choice == yes:
                self.params['check'] = False
        self.N = len(self.circles)
    
    def extract_data(self):
        if not os.path.exists(self.picklepath):
            os.makedirs(self.picklepath)
        if os.path.isfile(self.picklepath) == False or self.overwrite == True:
            data = [self.data.raw,self.data.timestamps] #[[x,y,r],t]
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