from tracking_class import Tracking
import matplotlib.pyplot as plt
import numpy as np

# this script loads circle data and plots something


root = '/Users/Jonas/OneDrive - UvA/PhD'
path = '/video tracking'
name  ='/Walking_slope_2'
ext = '.mp4'


filepath = [root,path,name,ext]
f = Tracking(filepath=filepath)

t,data = f.load_data()


x,y,r = data[:,:,0],data[:,:,1],data[:,:,2]
print(np.shape(t),np.shape(x))
plt.scatter(x,y)
plt.show()

