#Generate excitations, energies and rdms for excitations built on CHK state
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from crystal2pyscf import crystal2pyscf_cell
from methods import calcIAO, rdmIAO, hIAO, group
from basis import basis, minbasis, basis_order
import seaborn as sns
sns.set_style("white")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import find_connect
from dft_modelp import genex
from sklearn.decomposition import PCA 
from sklearn import preprocessing
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import OrthogonalMatchingPursuit

#Load data
fname="dft_model_iao3_s_16_10_0.7.pickle"
df=pd.read_pickle(fname)
df["E"]*=27.2
df["E"]-=df["E"][0]

#Data matrix
X=df.drop(columns=["E"])
zind=[]
for i in list(X):
  if(("3s" in i) or ("3p" in i) or ("5s" in i) or ("cu4s_1-cu4s_1-0" in i)): 
    zind.append(i)
X=X.drop(columns=zind)
y=df["E"]

#Rank checking
u,s,v=np.linalg.svd(X)
print(s)
print(X.shape)
print(np.linalg.matrix_rank(X,tol=1e-6))

#OMP
cscores=[]
cscores_err=[]
scores=[]
conds=[]
nparms=[]
#for i in range(1,X.shape[1]+1):
for i in range(41,42):
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
  
  plt.title(fname)
  plt.xlabel("Predicted energy (eV)")
  plt.ylabel("DFT Energy (eV)")
  plt.plot(omp.predict(X),y,'og')
  plt.plot(y,y,'b-')
  plt.savefig(fname.split("p")[0][:-1]+".fit_fix.pdf",bbox_inches='tight')
  #plt.plot(np.arange(len(omp.coef_)),omp.coef_,'o-',label="Nparms= "+str(i))
'''
plt.axhline(0,color='k',linestyle='--')
plt.xticks(np.arange(len(list(X))),list(X),rotation=90)
plt.title(fname)
plt.ylabel("Parameter (eV)")
plt.legend(loc='best')
plt.savefig(fname.split("p")[0][:-1]+".omp_fix.pdf",bbox_inches='tight')
'''
