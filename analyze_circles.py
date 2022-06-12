from tracking_class import Tracking
import matplotlib.pyplot as plt
import numpy as np

# this script loads circle data and plots something


root = '/Users/Jonas/OneDrive - UvA/PhD'
path = '/video tracking'
name  ='/Basler_acA640-750um__23036100__20220610_183814017_1186'
ext = '.tiff'


filepath = [root,path,name,ext]
f = Tracking(filepath=filepath)

t,data = f.load_data()


x,y,r = data[:,:,0],data[:,:,1],data[:,:,2]
print(np.shape(t),np.shape(x))
plt.scatter(x[0,:],y[0,:])
plt.show()

