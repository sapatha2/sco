import numpy as np 
import pandas as pd
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error, r2_score

#Sum over NN 
#Divide by two 
#For the NN -> -1/4 for each singlet, 1/4 for each triplet
#       since Si Sj just gives -1/4 of itself and 1/2 for the switch

sigJ=np.array([-2, 0, 0, 2])
E_PBE0=np.array([-9.2123749620223E+02, -9.2122638412999E+02, -9.2122636552756E+02, -9.2121137910381E+02])
E0=E_PBE0[0]
E_PBE0-=E0
E_PBE0*=27.2

fit_PBE0 = linear_model.LinearRegression(fit_intercept=True)
fit_PBE0.fit(sigJ[:,np.newaxis],E_PBE0)

E_PBE0_pred=fit_PBE0.predict(sigJ[:,np.newaxis])

print(fit_PBE0.coef_[0],r2_score(E_PBE0,E_PBE0_pred))
plt.plot(E_PBE0,E_PBE0_pred,'bo')
plt.plot(E_PBE0,E_PBE0,'g')

plt.ylabel("E(pred), eV)")
plt.xlabel("E-E(CHK), eV")

plt.title("J="+str(fit_PBE0.coef_[0])+", R2="+str(r2_score(E_PBE0,E_PBE0_pred)))
plt.show()

