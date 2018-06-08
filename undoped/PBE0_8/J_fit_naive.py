import numpy as np 
import pandas as pd
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error, r2_score

#PBE0_8 BFD CUTOFF
#sigJ=np.array([-4,-2,0,0,0,0,4])
#E_PBE0=np.array([-1.8424749919822E+03,-1.8424644227349E+03,-1.8424527681039E+03 ,-1.8424548946115E+03,-1.8424530584090E+03,-1.8424527307933E+03,-1.8424227577357E+03])

#PBE0_8 BFD 0.2
sigJ=np.array([-4,-2,0,0,0,0,4])
E_PBE0=np.array([-1.8420885803128E+03,-1.8420775305468E+03 ,-1.8420649624789E+03,-1.8420675966169E+03,-1.842065281760E+03,-1.8420655625350E+03,-1.8420340646210E+03])

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

