from tracking_class import Tracking
import matplotlib.pyplot as plt
import numpy as np
import csv,os
# this script loads circle data and plots something
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import matplotlib.cm as cm
class analyze(object):
    def __init__(self,data,ti=0,tf=10000):
        self.plot=False
        data,t=data
        self.N = np.shape(data)[1]
        pos = data[:,:,:2]
        self.r_mean = np.mean(data[:,:,2])
        self.t=np.asarray(t[ti:tf])

        xcom = self.compute_pos_CoM(pos[0])
        theta = self.compute_theta_CoM(xcom)
        circles = np.c_['-1',pos[0],theta]
        pos = pos[:,np.argsort(circles[:,-1])]

        th = []
        for step in pos:
            th.append(self.compute_theta_me(step))
            
        self.th=np.asarray(th)
        self.get_angles(filepath[:2],np.asarray(th),ti,tf)
        self.pos_ = - pos[:,:,0] + 1j *pos[:,:,1] #minus sign to change direction

        pass
    
    
    def get_vt(self):
        self.pos_ /= self.r_mean #convert to cm, rmean is 1 cm, (check if stickers are 1cm)
        poscom_ = np.mean(self.pos_,axis=1)
        poscom_ -= poscom_[0]
        slope,b = np.polyfit(np.real(poscom_),np.imag(poscom_),1)
        poscom_ *= np.exp(-1j*np.angle(1+ 1j*slope))
        xcom = np.real(poscom_)
        return slope,self.t,xcom
        
        
        
        
    
    def compute_pos_CoM(self,data):
        data = np.asarray(data)
        xcm = np.sum(data[:,0])/self.N
        ycm = np.sum(data[:,1])/self.N
        return np.asarray([data[:,0] - xcm, data[:,1] - ycm]).transpose()
    def compute_theta_CoM(self,data):
        return np.arctan2(data[:,1],data[:,0])
    def compute_theta_me(self,step):
        reltheta = []
        for j,unit in enumerate(step):
            
            if j==self.N - 1:
                fwd = 0
            else:
                fwd = j+1
            a = np.sqrt((unit[0] - step[j-1][0])**2 + (unit[1] - step[j-1][1])**2)
            b = np.sqrt((step[fwd][0] - step[j-1][0])**2 + (step[fwd][1] - step[j-1][1])**2)
            c = np.sqrt((unit[0] - step[fwd][0])**2 + (unit[1] - step[fwd][1])**2)
            reltheta.append(np.arccos((c**2+a**2 -b**2)/(2*c*a)))
        return np.asarray(reltheta)

    def get_angles(self,outputpath,th,ti,tf,write=True):
        a = th - (self.N-2)/self.N*np.pi
        th1 = a[ti:tf,1]
        th3= a[ti:tf,3]
        th5= a[ti:tf,5]
        if self.plot:
            for ni,i in enumerate(a[0,:6]):
            
                plt.scatter(self.t,th1,s=1)
                plt.scatter(self.t,th3,s=1)
                plt.scatter(self.t,th5,s=1)
            plt.show()
        c = a
        if write == True:
            if not os.path.exists(figpath):
                os.makedirs(figpath)
            writer = csv.writer(open(figpath + f'angledata {name[1:]}.csv', 'w'))
            for row in c:
                writer.writerow(row)
        return th1,th3,th5

    def get_colors(self,th1,th2,th3):
        tr,s1,s2=self.get_basis(th1,th2,th3)
        phase=((np.angle(s1+1j*s2))%(2*np.pi))/(2*np.pi)

        clist=[]
        for p in phase:    
            rval=255
            gval=(1-p)*218
            bval=219-(219-5)*p
            clist+=[(rval/255,gval/255,bval/255)]
        return clist

    def get_basis(self,th1,th2,th3):
        tri=(th1+th2+th3)/np.sqrt(3)
        s1=(th1+th2-2*th3)/np.sqrt(6)
        s2=(th1-th2)/np.sqrt(2)
        return tri,s1,s2

    def plot_cycle(self,ti,tf):
        t=self.t
        th=self.th
        b =[1,5,3] 
        
        fig = plt.figure(figsize =((12,8)))
        ax = fig.add_subplot(111)
        th = th - 4 * np.pi / 6
        th1_r = th[ti:ti+tf,b[0]]/np.pi
        th2_r = th[ti:ti+tf,b[1]]/np.pi
        th3_r = th[ti:ti+tf,b[2]]/np.pi
        tr, s1,s2=self.get_basis(th1_r,th2_r,th3_r)
        clist=self.get_colors(th1_r,th2_r,th3_r)
        
        q = ax.plot(s1,s2,linewidth=3,c="black",zorder=1)
        p = ax.scatter(s1,s2,c=clist,s=100,zorder=2)
        f = ax.fill(s1,s2,c="lightgray",alpha=1,edgecolor="None",zorder=0)
        ax.set_xlim([-0.3,0.3])
        ax.set_ylim([-0.3,0.3])
        ax.set_aspect('equal')
        ax.axhline(0,c="gray",zorder=-2,ls="--")
        ax.axvline(0,c="gray",zorder=-2,ls="--")
        arind=0
        ar_kwargs={"color": "gray","arrowstyle": "Simple,tail_width=2, head_width=15,head_length=15"}
        ar1=mpatches.FancyArrowPatch((0.9*s1[arind],0.9*s2[arind]),(0,0),**ar_kwargs)
        ax.add_patch(ar1)

        rad=0.3
        agl=2*np.pi/3
        ar2=mpatches.FancyArrowPatch( (rad,0),(rad*np.cos(agl),rad*np.sin(agl)),connectionstyle="arc3,rad=0.5",**ar_kwargs)
        ax.add_patch(ar2)
        fig.savefig(figpath + f'cycleplot {name[1:]}.pdf')

    def kymo(self):
        fig,ax = plt.subplots(1,figsize=(2.5,3))
        tmin = self.t[0]
        tmax = self.t[-1]
        xmin=0
        xmax= np.shape(self.th)[1]
        ax.imshow(self.th,aspect='auto',origin='lower',extent = [xmin,xmax,tmin,tmax],cmap = 'PiYG')
        ax.set_xlabel('units')
        ax.set_ylabel('time (s)')
        plt.tight_layout()
        fig.savefig(figpath + f'kymo {name[1:]}.pdf')
        # return fig

    def get_phasevelocity(self):
        dth_max = (np.argmax(self.th,axis=1)[-200:-100])%float(6)
        dth_max = np.unwrap(dth_max, period=6)
        
        # fig,ax = plt.subplots(1,figsize=(2.5,3))
        # ax.scatter(np.arange(len(dth_max)),dth_max)
        # plt.show()
        # 
        
        vphase,b = np.polyfit(self.t[-200:-100],dth_max,1)
        # fig,ax = plt.subplots(1,figsize=(2.5,3))
        # ax.scatter(self.t[:-100],(np.argmax(self.th,axis=1)[:-100])%float(6))
        # ax.plot(self.t,vphase*self.t + b)
        # plt.show()

        return np.abs(vphase)


# hexagon limit cycle analysis
# root = '/Users/Jonas/OneDrive - UvA/video data/fatboy6/220714 - overdamped underdamped hexagon'
# path = '/overdamped'
# name  ='/overdamped eps=7'
# ext = '.mov'
# filepath = [root,path,name,ext]
# figpath = root+path+'/figures/'
# f = Tracking(filepath=filepath)
# data= f.load_data()
# A = analyze(data)
# fig,ax = plt.subplots(1,figsize=(2.5,3))
# A.plot_cycle(222,500)




# locomotino experiment analysis
root = '/Users/Jonas/OneDrive - UvA/video data/fatboy12/220714 - acceleration versus slope'
drive = '/odd'
path = '/1.5 slope'
name  ='/DSC_1156'
ext = '.mov'


# filepath = [root+drive,path,name,ext]
# figpath = root+drive+path+'/figures/'
# f = Tracking(filepath=filepath)
# data= f.load_data()
# A = analyze(data)
# A.kymo()



odd=True
TM=True
get_solidbody_plots=True
if get_solidbody_plots:
    fig,ax = plt.subplots(4,2,figsize=(16,16))
    bx = ax.flatten()[:4]
    ax = ax.flatten()[4:]

    sfig,sax = plt.subplots(1,2,figsize=(6,3),dpi=150)
    fig2,ax2 = plt.subplots(1,2,figsize=(6,3),dpi=150)
    colors = cm.rainbow(np.linspace(0, 1,25))
    fig3,ax3 = plt.subplots(1,figsize=(3,3),dpi=150)


    for d in ['/odd','/TM']:
        slopes=[]
        vss = []
        ts = []
        afits = []
        vphases=[]
        i=0
        root = '/Users/Jonas/OneDrive - UvA/video data/fatboy12/220714 - acceleration versus slope'
        for n,[subdir, dirs, files] in enumerate(os.walk(root+d)):

            for m,file in enumerate(files):
                
                if file[-4:]=='.mov':
                    root = '/'.join(subdir.split('/')[:-1])
                    path = '/'+subdir.split('/')[-1]
                    slope_msrd = float(path.split(' ')[0][1:])

                    name  ='/' +file[:-4]
                    ext = '.mov'
                    
                    
                    filepath = [root,path,name,ext]
                    figpath = root+path+'/figures/'

                
                    f = Tracking(filepath=filepath)
                    data= f.load_data()
                    A = analyze(data)
                    A.kymo()
                    vphase = A.get_phasevelocity()
                    
                    slope,t,xcom =  A.get_vt()
                    
                    start_thr = 0.2
                    idx_start = np.argmax(xcom>start_thr) - 10

                    t=t[idx_start:]
                    xcom=xcom[idx_start:]
                    
                    xcomspl = UnivariateSpline(t, xcom)
                    vcomspl = xcomspl.derivative()
                    
                    
                    acctf = 150
                    aspl,_ = np.polyfit(t[:acctf],vcomspl(t[:acctf]),1)
                    
                    def poly(x,coeff):
                        x=np.asarray(x)
                        r = np.asarray([a*x**float(i) for i,a in enumerate(coeff)])
                        return np.sum(r,axis=0)
                    def xfitfunc(x,a,b):
                        return a*x**2 + b*x**2
                    def vfitfunc(x,a,b):
                        return 2*a*x+b
                    
                    # 
                    coeff = np.polyfit(t,xcom,3)
                    posfit = poly(t,np.flip(coeff))
                    
                    vfit = poly(t,np.asarray([(i+1)*c for i,c in enumerate(np.flip(coeff)[1:])]))
                    idx_av= 20
                    dxcom = (xcom - np.roll(xcom,idx_av))/(t[idx_av]-t[0])
                    dxcom_max = np.argmax(dxcom)
                    
                    
                    tfit = np.asarray(t[:np.argmax(vfit)])
                    try:
                        tfit -= tfit[0]
                        posfit = posfit[:np.argmax(vfit)]
                        xcomfit = xcom[:np.argmax(vfit)]
                        
                        xcomparabola = xcom[:np.argmax(vfit)]
                        
                        popt, pcov = curve_fit(xfitfunc,tfit, xcomparabola)
                        xcomparabola = xfitfunc(tfit,*popt)
                        vcomparabola = vfitfunc(tfit,*popt)
                        
                        vfit = vfit[:np.argmax(vfit)]
                        afit,_ = np.polyfit(t[:acctf],vfit[:acctf],1)
                        bx[0].scatter(tfit, xcomfit, color = colors[int(2*slope_msrd)],s=5)
                        bx[1].scatter(t[:dxcom_max -idx_av], dxcom[idx_av:dxcom_max], color = colors[int(2*slope_msrd)],s=5)
                        bx[2].scatter(tfit, posfit, color = colors[int(2*slope_msrd)],s=5)
                        bx[3].scatter(tfit, vfit, color = colors[int(2*slope_msrd)],s=5)

                        bx[0].set_title('raw data')
                        bx[1].set_title('raw data')
                        bx[2].set_title('polyfit')
                        bx[3].set_title('polyfit')
                        ax[0].set_title('spline')
                        ax[1].set_title('spline')
                        ax[2].set_title('parametric')
                        ax[3].set_title('parametric')
                        
                        
                        bx[0].set_xlabel('time (s)')
                        bx[1].set_xlabel('time (s)')
                        bx[0].set_ylabel('Centre of mass position (cm)')
                        bx[1].set_ylabel('Centre of mass velocity (cm/s)')
                        bx[2].set_xlabel('time (s)')
                        bx[3].set_xlabel('time (s)')
                        bx[2].set_ylabel('Centre of mass position (cm)')
                        bx[3].set_ylabel('Centre of mass velocity (cm/s)')
                        # 
                        
                        
                        ax[0].set_xlabel('time (s)')
                        ax[1].set_xlabel('time (s)')
                        ax[2].set_xlabel('slope $\Delta$ (%)')
                        ax[3].set_xlabel('slope $\Delta$ (%)')
                        ax[0].set_ylabel('Distance travelled (cm)')
                        ax[1].set_ylabel('Centre of mass velocity (cm/s)')
                        ax[2].set_ylabel('Steady state velocity (cm/s)')
                        ax[3].set_ylabel('Initial acceleration (cm/s^2)')
                        
                        sax[0].set_ylabel('Steady state velocity (cm/s)')
                        sax[1].set_ylabel('Initial acceleration (cm/s^2)')
                        sax[0].set_xlabel('slope $\Delta$ (%)')
                        sax[1].set_xlabel('slope $\Delta$ (%)')
                        
                        ax3.set_xlabel('slope $\Delta$ (%)')
                        ax3.set_ylabel('steady state phase velocity $v_{\\phi}$')
                    
                    except IndexError:
                        pass
                    
                    print(slope_msrd)
                    
                    if slope_msrd==1.5:
                        if d=='/odd' and odd:
                            print('he')
                            ax2[0].set_xlabel('time (s)')
                            ax2[0].set_ylabel('Centre of mass velocity (cm/s)')
                            traw = np.asarray(t[:dxcom_max -idx_av]) - t[0]
                            
                            length = len(traw)
                            
                            vt = dxcom[idx_av:dxcom_max]
                            ax2[0].scatter(traw, vt,color='r',s=5,label='odd')
                            odd=False

                        if d=='/TM' and TM:
                            
                            ax2[0].set_xlabel('time (s)')
                            ax2[0].set_ylabel('Centre of mass velocity (cm/s)')
                            traw = np.asarray(t[:dxcom_max -idx_av]) - t[0]
                            vt =dxcom[idx_av:dxcom_max]
                            
                            ax2[0].scatter(traw[:length], vt[:length],color='b',s=5,label='TM')
                            TM=False
                            ax2[0].legend()

                            
                    
                    
                    
                    i+=1
                    
                    # 
                    # 
                    vphases.append(vphase)
                    slopes.append(slope_msrd)
                    vss.append(np.max(vfit))
                    afits.append(afit)
                    ts.append(tfit)



                                
        vals, idx_start, count = np.unique(slopes, return_counts=True, return_index=True)
        ind = np.argsort(slopes)
        print(vals)
        if d=='/odd':
            col = 'r'
        if d=='/TM':
            col = 'b'

        cumcount = np.asarray([0,*np.cumsum(count)])
        vavg=[]
        aavg =[]
        vstd=[]
        astd=[]
        vphaseavg=[]
        vphasestd=[]
        vss = np.asarray(vss)
        afits = np.asarray(afits)
        vphases=np.asarray(vphases)

        for n,[v,c,cv] in enumerate(zip(vals,count,cumcount)):
            avind = ind[cumcount[n]:cumcount[n+1]]

            vavg.append(np.average(vss[avind]))
            vstd.append(np.std(vss[cv:cumcount[n+1]]))
            aavg.append(np.average(afits[cv:cumcount[n+1]]))
            astd.append(np.std(afits[cv:cumcount[n+1]]))
            vphaseavg.append(np.average(vphases[avind]))
            vphasestd.append(np.std(vphases[cv:cumcount[n+1]]))


        
        
        print(np.shape(vphaseavg))
        print(np.shape(vavg))
        print(np.shape(vstd))
        
        ax3.errorbar(vals,vphaseavg,yerr = vphasestd,c=col,label=d[1:],capsize = 3,marker='^', fmt='.')
        
        sax[0].errorbar(vals,vavg,yerr = vstd,c=col,label=d[1:],capsize = 3,marker='^', fmt='.')
        sax[1].errorbar(vals,aavg,yerr = astd,c=col,label=d[1:],capsize = 3,marker = '^', fmt='.')
        sax[0].legend()
        sax[1].legend()
        ax3.legend()
                    
                    
                    
                    
        higherslope = [12,13.5,15,16.5,18,19.5]
        sax[0].scatter(higherslope,[0. for i in higherslope],c='r',marker = '^')
        sax[1].scatter(higherslope,[0. for i in higherslope],c='r',marker = '^')
        ax3.scatter(higherslope,[0. for i in higherslope],c='r',marker = '^')            


    sfig.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    sfig.savefig(root+'/figures/' + f'vplot.pdf')
    fig2.savefig(root+'/figures/' + f'single_vt.pdf')
    fig3.savefig(root+'/figures/' + f'vphase_plot.pdf')
    plt.show()


                






            
                

