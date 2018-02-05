import numpy as np 
import pandas as pd
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error, r2_score

def ECHK(s):
  return -4*s*(s+1)
def ECOL(s):
  return 2*s*(2*s+1)-4*s*(s+1)
def EFM(s):
  return 2*2*s*(2*s+1)-4*s*(s+1)

S_PBE0=[0.5853661,0.6442761,0.7176682]
S_HSE=[0.5788231,0.6385404,0.7131834]
X_basic=np.array([-3,-1,1]) #If it were spin-1/2 on each atom; undoped CHK, COL/FLP, FM
X_PBE0=np.array([ECHK(S_PBE0[0]),ECOL(S_PBE0[1]),EFM(S_PBE0[2])]) #Based on onsite spin from PBE0 calculations
X_HSE=np.array([ECHK(S_HSE[0]),ECOL(S_HSE[1]),EFM(S_HSE[2])]) #Based on onsite spin from PBE0 calculations
y_PBE0=np.array([-9.2123749620223E+02,-9.2122638412999E+02,-9.2121137910381E+02])*27.2
y_HSE=np.array([-9.2122881518088E+02,-9.2121750188336E+02,-9.2120231036158E+02])*27.2
y_PBE0-=y_PBE0[0]
y_HSE-=y_HSE[0]

fit_PBE0_basic = linear_model.LinearRegression(fit_intercept=True)
fit_PBE0_basic.fit(X_basic[:,np.newaxis],y_PBE0)
fit_PBE0 = linear_model.LinearRegression(fit_intercept=True)
fit_PBE0.fit(X_PBE0[:,np.newaxis],y_PBE0)
'''
print(fit_PBE0_basic.coef_,fit_PBE0_basic.intercept_,r2_score(y_PBE0,fit_PBE0_basic.predict(X_basic[:,np.newaxis])))
print(fit_PBE0.coef_,fit_PBE0.intercept_,r2_score(y_PBE0,fit_PBE0.predict(X_PBE0[:,np.newaxis])))

#COEFF         INTERCEPT      R2
#[ 0.17759627] 0.515140749905 0.992648383155
#[ 0.1229954]  0.458098005928 0.999958634484
'''
fit_HSE_basic = linear_model.LinearRegression(fit_intercept=True)
fit_HSE_basic.fit(X_basic[:,np.newaxis],y_HSE)
fit_HSE = linear_model.LinearRegression(fit_intercept=True)
fit_HSE.fit(X_HSE[:,np.newaxis],y_HSE)
'''
print(fit_HSE_basic.coef_,fit_HSE_basic.intercept_,r2_score(y_HSE,fit_HSE_basic.predict(X_basic[:,np.newaxis])))
print(fit_HSE.coef_,fit_HSE.intercept_,r2_score(y_HSE,fit_HSE.predict(X_HSE[:,np.newaxis])))
#COEFF         INTERCEPT      R2
#[ 0.18023277] 0.523117030409 0.992913912945
#[ 0.12658084] 0.465161168737 0.999897290319
'''

plt.plot([-0.01,-0.01,-0.01],y_PBE0,'bo',label='DFT energy')
plt.plot([-0.01,-0.01,-0.01],fit_PBE0_basic.predict(X_basic[:,np.newaxis]),'ro',label='Basic pred')
plt.plot([-0.01,-0.01,-0.01],fit_PBE0.predict(X_PBE0[:,np.newaxis]),'go',label='Better pred')

plt.plot([0.01,0.01,0.01],y_HSE,'bo')
plt.plot([0.01,0.01,0.01],fit_HSE_basic.predict(X_basic[:,np.newaxis]),'ro')
plt.plot([0.01,0.01,0.01],fit_HSE.predict(X_HSE[:,np.newaxis]),'go')

plt.title("DFT linear regression x=0")
plt.ylabel(r"$E-E_{CHK}$,eV")
plt.legend(loc=1)
plt.show()
