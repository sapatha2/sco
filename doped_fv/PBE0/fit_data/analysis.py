#Generate excitations, energies and rdms for excitations built on CHK state
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from crystal2pyscf import crystal2pyscf_cell
from methods import genex
from basis import basis, minbasis, basis_order
import seaborn as sns
sns.set_style("white")
from pyscf import lo
from functools import reduce 
from pyscf2qwalk import print_qwalk_pbc
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import OrthogonalMatchingPursuit
from statsmodels.sandbox.regression.predstd import wls_prediction_std

#Load data frame
df=pd.read_pickle("singlesR_0.7_fix.pickle")

#plt.hist(df["E"]-df["E"][0],bins=20)
#plt.show()

#LinReg
y=df["E"]
y-=y[0]
#X=df.drop(columns=["E"])
ind=[]
for x in list(df):
  if(("3s" in x) or ("3p" in x) or ("3dxy" in x) or ("3dxz" in x) or ("3dyz" in x) or ("2pz" in x) or ("2ppi" in x)): ind.append(x)
  if(("4s" in x) or ("3dz2" in x)): ind.append(x)
  if(("2s" in x)): ind.append(x)
#ind+=["o2psig-o2s-0.00","o2s_u","c3dx2y2-o2s-0.25"]
X=df.drop(columns=ind)
print(list(X))
sns.pairplot(X)
plt.show()
exit(0)

X=sm.add_constant(X)

#sns.pairplot(X)
#plt.show()

#OLS
#Fit to smaller
res_ols=sm.OLS(y,X).fit()
print(res_ols.summary())

#Predict on bigger!
X=df.drop(columns=["E"])
X=X.drop(columns=ind)
X=sm.add_constant(X)
y=df["E"]
y-=y[0]
plt.plot(y,res_ols.predict(X),'og')
plt.plot(y,y,'b-')
plt.xlabel("E [eV]")
plt.ylabel("Predicted E [eV]")
plt.show()

#WLS
'''
beta=5.
w=np.exp(-2*beta*y)
res_wls=sm.WLS(y,X,weights=w).fit()
print(res_wls.summary())
prstd, iv_l, iv_u = wls_prediction_std(res_wls)
plt.errorbar(y,res_wls.fittedvalues,yerr=(iv_u-iv_l)/2.,fmt='og')
plt.plot(y,y,'b-')
#plt.plot(y,iv_u,'g.')
#plt.plot(y,iv_l,'g.')
plt.ylim((min(y),max(y)))
plt.show()
'''

'''
#Rank checking
u,s,v=np.linalg.svd(X)
print(s)
print(X.shape)
print(np.linalg.matrix_rank(X,tol=1e-6))
'''

#OMP
'''
cscores=[]
cscores_err=[]
scores=[]
conds=[]
nparms=[]
for i in range(1,X.shape[1]+1):
  print("n_nonzero_coefs="+str(i))
  omp = OrthogonalMatchingPursuit(n_nonzero_coefs=i)
  omp.fit(X,y)
  nparms.append(i)
  scores.append(omp.score(X,y))
  tmp=cross_val_score(omp,X,y,cv=5)
  cscores.append(tmp.mean())
  cscores_err.append(tmp.std()*2)
  print("R2: ",omp.score(X,y))
  print("R2CV: ",tmp.mean(),"(",tmp.std()*2,")")
  ind=np.abs(omp.coef_)>0
  Xr=X.values[:,ind]
  conds.append(np.linalg.cond(Xr))
  print("Cond: ",np.linalg.cond(Xr))
  print(np.array(list(X))[ind])
  print(omp.coef_[ind])
'''  
'''
  plt.title(fname)
  plt.xlabel("Predicted energy (eV)")
  plt.ylabel("DFT Energy (eV)")
  plt.plot(omp.predict(X),y,'og')
  plt.plot(y,y,'b-')
  #plt.savefig(fname.split("p")[0][:-1]+".fit_fix.pdf",bbox_inches='tight')
  plt.plot(np.arange(len(omp.coef_)),omp.coef_,'o-',label="Nparms= "+str(i))
'''
'''
plt.axhline(0,color='k',linestyle='--')
plt.xticks(np.arange(len(list(X))),list(X),rotation=90)
plt.title(fname)
plt.ylabel("Parameter (eV)")
plt.legend(loc='best')
plt.savefig(fname.split("p")[0][:-1]+".omp_fix.pdf",bbox_inches='tight')
'''
