import numpy as np 
import pandas as pd
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error, r2_score

def sigJ(a,b,d,c):
  return 0.5*((a+b)*(a+b+1) + (b+c)*(b+c+1) + (c+d)*(c+d+1) +  (d+a)*(d+a+1)) - abs(a)*(abs(a)+1) - abs(b)*(abs(b)+1) - abs(c)*(abs(c)+1) - abs(d)*(abs(d)+1)

#print(sigJ(0.5015396,0.1592959,-0.4706253,0.5015402))
#print(sigJ(0.5209890,-0.0018191,0.6160942,-0.4347114))
#print(sigJ(0.4981368,0.4981459,0.4981262,0.4981324))

#CHK, COL, FM
sigJ_PBE0=np.array([sigJ(0.5015396,0.1592959,-0.4706253,0.5015402),sigJ(0.5209890,-0.0018191,0.6160942,-0.4347114),sigJ(0.4981368,0.4981459,0.4981262,0.4981324)])
sigJ_HSE=np.array([sigJ(0.4742836,0.1959011,-0.4567655,0.4742838),sigJ(0.5872262,-0.2385957,0.5872257,-0.2385895),sigJ(0.4965528,0.4965522,0.4965519,0.4965527)])

y_PBE0=np.array([-9.2092669002320E+02,-9.2092063265454E+02,-9.2091264054414E+02])*27.2
y_HSE=np.array([-9.2092109901659E+02,-9.2091332695333E+02,-9.2090837968879E+02])*27.2
y_PBE0-=y_PBE0[0]
y_HSE-=y_HSE[0]


fit_PBE0 = linear_model.LinearRegression(fit_intercept=True)
fit_PBE0.fit(sigJ_PBE0[:,np.newaxis],y_PBE0)
fit_HSE = linear_model.LinearRegression(fit_intercept=True)
fit_HSE.fit(sigJ_HSE[:,np.newaxis],y_HSE)

plt.plot([-0.01,-0.01,-0.01],y_PBE0,'bo',label='DFT energy')
plt.plot([-0.01,-0.01,-0.01],fit_PBE0.predict(sigJ_PBE0[:,np.newaxis]),'go',label='J pred')

plt.plot([0.01,0.01,0.01],y_HSE,'bo')
plt.plot([0.01,0.01,0.01],fit_HSE.predict(sigJ_HSE[:,np.newaxis]),'go')

plt.show()
