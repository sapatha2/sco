import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

'''
#undoped: J
E=(E-E[0])*27.2
sigJ=np.array([-2,0,2])
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
'''

#doped: J, bands and intercept model
E=np.array([-1.8418601953776E+03,-1.8418566128030E+03,-1.8418522579094E+03,-1.8418450389802E+03,-1.8418284707358E+03])
E=(E-E[0])*27.2
sigJ=np.array([0,0,0,-4,0])
sigJ=sigJ.reshape((len(sigJ),1))

regr = linear_model.LinearRegression()
regr.fit(sigJ, E)
J_only_pred = regr.predict(sigJ)
plt.subplot(222)
plt.title("DOPED Only J model; J="+str(regr.coef_[0])+", R^2="+str(r2_score(E,J_only_pred)))
plt.xlabel("E-E[FLP] (eV)")
plt.ylabel("Predicted E (eV)")
plt.plot(E,J_only_pred,'bs')
plt.plot(E,E,'b--')

TB=np.array([-4.5001188191477022,-4.534741615141737,-2.7558317335899405,-1.2826154840345547,-2.6418027304882523])
#TB=np.array([-2.2500594095738511,-2.0822537131125491,-1.7103427869666361,-0.4958248152005606,-1.6909919759294867])
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
