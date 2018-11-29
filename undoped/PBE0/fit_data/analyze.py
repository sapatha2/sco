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

df=pd.read_pickle("dft_model_iao3_16.pickle")
df["E"]*=27.2
df["E"]-=df["E"][0]

#plt.hist(df["E"])
#plt.show()

'''
#plt.plot(df.var().values,'o')
#plt.xticks(np.arange(len(list(df))),list(df),rotation=60)
#plt.show()

plt.matshow(df.corr(),vmin=-1,vmax=1,cmap=plt.cm.bwr)
plt.colorbar()
plt.xticks(np.arange(len(list(df))),list(df),rotation=90)
plt.yticks(np.arange(len(list(df))),list(df))
plt.show()
'''

#X=df.drop(columns=["E"])
#X=sm.add_constant(X)
'''
X=df[['o2s_1-o2s_1-0' ,'cu3dx2y2_1-cu3dx2y2_1-0', 'o2ppi_1-o2ppi_2-0',
 'cu3dxy_1-cu3dxy_1-0' ,'cu3dyz_1-cu3dyz_1-0', 'cu3dz2_1-cu3dz2_1-0',
 'cu3dxy_2-cu3dxy_2-0', 'cu4s_1-cu4s_4-0', 'cu4s_1-cu4s_2-0',
 'cu3dyz_2-cu3dyz_2-0', 'cu3dz2_2-cu3dz2_2-0', 'o2psig_1-o2psig_1-0',
 'o2pz_1-o2pz_1-0', 'cu4s_1-cu4s_1-0', 'cu4s_2-cu4s_2-0', 'o2ppi_1-o2ppi_5-0',
 'o2ppi_1-o2ppi_1-0', 'o2s_1-o2s_2-0' ,'cu4s_2-o2psig_2-0',
 'cu4s_1-o2psig_1-0', 'cu4s_1-o2s_1-0' ,'cu4s_2-o2s_2-0',
 'cu3dx2y2_2-o2psig_2-0', 'cu3dx2y2_2-o2s_2-0', 'o2psig_1-o2psig_3-0',
 'cu4s_1-o2s_2-0', 'cu4s_2-o2s_1-0', 'cu3dx2y2_1-o2psig_1-0',
 'cu4s_2-cu3dz2_2-0', 'cu4s_1-cu3dz2_1-0', 'cu3dz2_2-o2psig_2-0',
 'cu3dz2_1-o2psig_1-0', 'o2s_1-o2s_5-0', 'o2s_1-o2s_7-0',
 'cu3dxy_2-o2ppi_2-0', 'cu3dxy_1-o2ppi_1-0', 'cu4s_1-cu3dx2y2_2-0']]
y=df["E"]
model=sm.OLS(y,X).fit()
print(model.summary())
plt.plot(model.predict(X),y,'go')
plt.plot(y,y,'-')
plt.show()
exit(0)
'''

import sympy 
X=df.drop(columns=["E"])
zind=[]
for i in list(X):
  if(("3s" in i) or ("3p" in i) or ("5s" in i)): 
    zind.append(i)
X=X.drop(columns=zind)
X=sm.add_constant(X)
u,s,v=np.linalg.svd(X)
print(s)
print(X.shape)
print(np.linalg.matrix_rank(X,tol=1e-6))
#exit(0)

pca=PCA()
pca.fit(X)
#plt.plot(np.cumsum(pca.explained_variance_ratio_),'o-')
#plt.show()

'''
#Correlation matrix (parameters only)
df=pd.DataFrame(X,columns=labels)
mat=df.corr().values
ordering=find_connect.recursive_order(mat,tols=[1e-4,1e-3,0.2,0.3,0.4,0.5,0.7,0.9,1.0])
mat=mat[ordering][:,ordering]
plt.matshow(mat,vmax=1,vmin=-1,cmap=plt.cm.bwr)
plt.xticks(np.arange(len(labels)),labels[ordering],rotation=90)
plt.yticks(np.arange(len(labels)),labels[ordering])
plt.show()
exit(0)

#PCA
pca=PCA()
pca.fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_),'o-',label=str(c))
Xr=pca.transform(X)
for i in range(Xr.shape[0]):
  plt.plot(Xr[i,:],'o',label=str(c))
plt.show()
'''

###########################################################################################
#ANALYSIS
#Plot changes in number occupation

'''
n_labels=[]
n_psums=[]
for i in range(len(full_labels)):
  l=full_labels[i].split("-")
  if(l[0]==l[1]): 
    n_labels.append(full_labels[i])
    n_psums.append(full_psums[i])
n_psums=np.array(n_psums)
n_labels=np.array(n_labels)
print(n_psums.shape,n_labels.shape)

for i in range(n_psums.shape[1]):
  plt.plot(n_psums[:,i],'.')
plt.xticks(np.arange(n_labels.shape[0]),n_labels,rotation=90)
plt.show()
'''

#Plots changes in all parameters
#for i in range(psums.shape[1]):
#  plt.plot(psums[:,i],np.arange(psums[:,i].shape[0]),'bo')
'''
var=[]
for i in range(psums.shape[0]):
  var.append(np.var(psums[i,:]))
ind=np.argsort(var)
plt.plot(ind,np.array(var)[ind],'bo')
plt.yticks(var,labels)
plt.show()
'''

'''
#Plot fit from rotated H1
#Full parameters
full_pred=np.einsum('ji,j->i',full_psums,full_parameters)
a=(full_pred-full_pred[0])*27.2114
b=(e_list-e_list[0])*27.2114
print(len(full_parameters),r2_score(a,b))
plt.plot(b,a,'bo')

plt.plot(b,b,'-')
plt.ylabel("Predicted energy")
plt.xlabel("PBE0 eigenvalue differences")
plt.show()

#Select parameters
pred=np.einsum('ji,j->i',psums,parameters)
a=(pred-pred[0])*27.2114
b=(e_list-e_list[0])*27.2114
print(len(parameters),r2_score(a,b))
plt.plot(b,a,'go')
'''

#X=df.drop(columns=["E"])
#Constrain search to just pi, sigma, s, 3dx2y2, 3dz2, 4s
'''
jnd=[]
for i in list(X):
  if ("3p" in i) or ("3s" in i) or ("5s" in i):
    pass
  else:
    jnd.append(i)
X=X[jnd]
'''

#OMP
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import OrthogonalMatchingPursuit
cscores=[]
cscores_err=[]
scores=[]
conds=[]
nparms=[]
#Full OMP
#n_lines=X.shape[1]
#ax = plt.axes()
#ax.set_color_cycle([plt.cm.Blues(i) for i in np.linspace(0, 1, n_lines)])
y=df["E"]
#for i in range(1,X.shape[1]+1):
for i in range(37,42):
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
  #print("Cond: ",conds[i-1])
  print(np.array(list(X))[ind])
  #Low condition number, high R2
  #if((conds[-1] < 100000) and (tmp.mean()>0.90)): plt.plot(np.arange(len(omp.coef_)),omp.coef_,'o-')
  plt.plot(omp.predict(X),y,'og')
  plt.plot(y,y,'b-')
  plt.show()
#plt.errorbar(np.log(conds),cscores,yerr=cscores_err,fmt='gs')
#plt.show()

#plt.plot(np.arange(len(omp.coef_)),parameters*27.2114,'k*-')
#plt.axhline(0,color='k',linestyle='--')
#plt.xticks(np.arange(len(list(X))+1),["E"]+list(X),rotation=60)
#plt.show()
'''
plt.subplot(131)
plt.plot(nparms,scores,'o')
plt.subplot(132)
plt.errorbar(nparms,cscores,yerr=cscores_err,fmt='gs')
plt.subplot(133)
plt.plot(nparms,conds,'r*')
plt.show()
'''

'''
#Restricted OMP
n_lines=X.shape[1]
ax = plt.axes()
ax.set_color_cycle([plt.cm.bwr(i) for i in np.linspace(0, 1, n_lines)])
for i in range(35,43):
  omp = OrthogonalMatchingPursuit(n_nonzero_coefs=i,fit_intercept=True)
  omp.fit(X,y)
  print(omp.score(X,y))
  plt.plot(np.arange(len(omp.coef_)),omp.coef_,'o-')
plt.plot(np.arange(len(omp.coef_)),parameters*27.2114,'k*-')
plt.axhline(0,color='k',linestyle='--')
plt.xticks(np.arange(len(labels)),labels,rotation=60)
plt.show()
'''
