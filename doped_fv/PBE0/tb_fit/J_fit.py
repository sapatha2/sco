import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

#undoped: J
E=np.array([-9.2123749620223E+02,-9.2122638412999E+02,-9.2121137910381E+02])
E=(E-E[0])*27.2
sigJ=np.array([-3,-1,1])
sigJ=sigJ.reshape((3,1))

regr = linear_model.LinearRegression()
regr.fit(sigJ, E)
J_only_pred = regr.predict(sigJ)

plt.subplot(221)
plt.title("UNDOPED Only J model; J="+str(regr.coef_[0])+", R^2="+str(r2_score(E,J_only_pred)))
plt.xlabel("E-E[CHK] (eV)")
plt.ylabel("Predicted E (eV)")
plt.plot(E,J_only_pred,'bs')
plt.plot(E,E,'b--')


#doped: J, bands and intercept model
E=np.array([-9.2092669022287E+02,-9.2091969201655E+02,-9.2091264054414E+02])
E=(E-E[0])*27.2
sigJ=np.array([-1,-1,1])
sigJ=sigJ.reshape((3,1))

regr = linear_model.LinearRegression()
regr.fit(sigJ, E)
J_only_pred = regr.predict(sigJ)

plt.subplot(222)
plt.title("DOPED Only J model; J="+str(regr.coef_[0])+", R^2="+str(r2_score(E,J_only_pred)))
plt.xlabel("E-E[FLP] (eV)")
plt.ylabel("Predicted E (eV)")
plt.plot(E,J_only_pred,'bs')
plt.plot(E,E,'b--')

TB=np.array([-2.0190016193597979,-1.8788794855673803,-1.9198682312763093])
E-=TB
regr = linear_model.LinearRegression()
regr.fit(sigJ, E)
J_only_pred = regr.predict(sigJ)

plt.subplot(224)
plt.title("DOPED TB(t+K) + J model; J="+str(regr.coef_[0])+", R^2="+str(r2_score(E,J_only_pred)))
plt.xlabel("E-E[FLP] (eV)")
plt.ylabel("Predicted E (eV)")
plt.plot(E+TB,J_only_pred+TB,'gs')
plt.plot(E+TB,E+TB,'g--')

plt.suptitle("PBE0")
plt.show()
