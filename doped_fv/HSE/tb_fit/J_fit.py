import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

#undoped: J
E=np.array([-9.2122881518088E+02,-9.2121750188336E+02,-9.2120231036158E+02])
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
E=np.array([-9.2092109920075E+02,-9.2091332695333E+02,-9.2090837968879E+02])
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

TB=np.array([-1.6102647673579904,-1.4878706184425667,-1.5950941225625459])
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

plt.suptitle("HSE")
plt.show()
