import numpy as np 
import pandas as pd
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error, r2_score

sigJ=np.array([-4,-2,0,0,0,0,4])
E_PBE0=np.array([-1.8424576299449E+03,-1.8424470530096E+03,-1.8424350036693E+03,-1.8424356023510E+03,-1.8424356023510E+03,-1.8424351814538E+03,-1.8424046204551E+03])

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

