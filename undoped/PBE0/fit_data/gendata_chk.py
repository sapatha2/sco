#Generate excitations, energies and rdms for excitations built on CHK state
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from crystal2pyscf import crystal2pyscf_cell
from pyscf2qwalk import print_qwalk_mol
from methods import calcIAO, rdmIAO, hIAO, genex, data_from_ex, getn, nsum, Usum 
from methods import ts, tsig, tpi, tz, too
from basis import basis, minbasis, basis_order
import seaborn as sns
sns.set_style("white")
from sklearn.linear_model import LinearRegression
import find_connect

###########################################################################################
#Build IAO 
direc="../CHK"
occ=[i for i in range(72)]
cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=direc+"/GRED.DAT",kred=direc+"/KRED.DAT",totspin=0)
a=calcIAO(cell,mf,minbasis,occ)

'''
h=hIAO(mf,a)
w,vr=np.linalg.eigh(h[0])
plt.plot(mf.mo_energy[0][0][:len(w)],'go',label='pbe0')
plt.plot(w,'bo',label='rot')
plt.ylabel("E (eV)")
plt.xlabel("Eigenvalue")
plt.legend(loc=2)
plt.show()
exit(0)
'''

#Build excitations
'''
occ=np.arange(24,66)
virt=np.arange(66,72)
ex='singles'
ex_list=genex(mf.mo_occ,[occ,occ],[virt,virt],ex=ex)
e_list,dm_list=data_from_ex(mf,a,ex_list)
print(ex_list.shape,e_list.shape,dm_list.shape)
print(e_list)
plt.scatter(np.arange(505),e_list-e_list[0])
plt.show()
exit(0)
'''

#Build excitations
occ={"../CHK":[np.arange(24,66),np.arange(24,66)],
     "../COL":[np.arange(24,66),np.arange(24,66)],
     "../FLP":[np.arange(24,67),np.arange(24,65)],
     "../FM": [np.arange(24,68),np.arange(24,64)]}
virt={"../CHK":[np.arange(66,72),np.arange(66,72)],
      "../COL":[np.arange(66,72),np.arange(66,72)],
      "../FLP":[np.arange(67,72),np.arange(65,72)],
      "../FM": [np.arange(68,72),np.arange(64,72)]}
totspin={"../CHK":0,"../COL":0,"../FLP":2,"../FM":4}
E0={"../CHK":-9.2123749620223E+02,
"../COL":-9.2122638412999E+02,"../FLP":-9.2122636552756E+02,
"../FM":-9.2121137910381E+02}
e_list=[]
dm_list=[]
for direc in ["../CHK","../COL"]:
  print(direc)
  ex='singles'
  cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=direc+"/GRED.DAT",kred=direc+"/KRED.DAT",totspin=totspin[direc])
  ex_list=genex(mf.mo_occ,occ[direc],virt[direc],ex=ex)
  tmp_e,tmp_dm=data_from_ex(mf,a,ex_list)
  tmp_e+=(E0[direc]-E0["../CHK"])
  e_list.append(list(tmp_e))
  dm_list.append(list(tmp_dm))
e_list=np.concatenate(e_list)
dm_list=np.concatenate(dm_list)

'''
#Trace
tr=np.einsum('ijkk->ij',dm_list)
tr=tr[:,0]+tr[:,1]
plt.plot(tr,'o')
plt.ylabel("Trace")
plt.xlabel("Excited state")
plt.show()
'''

#Analysis
n=getn(dm_list)
nsum=nsum(n)
Usum=Usum(n)

ts4s1=ts(dm_list,"4s",1)
ts4s2=ts(dm_list,"4s",2)
ts3dz21=ts(dm_list,"3dz2",1)
ts3dz22=ts(dm_list,"3dz2",2)
ts3d1=ts(dm_list,"3d",1)
ts3d2=ts(dm_list,"3d",2)

tsig4s1=tsig(dm_list,"4s",1)
tsig4s2=tsig(dm_list,"4s",2)
tsig3dz21=tsig(dm_list,"3dz2",1)
tsig3dz22=tsig(dm_list,"3dz2",2)
tsig3d1=tsig(dm_list,"3d",1)
tsig3d2=tsig(dm_list,"3d",2)

tp3dxy1=tpi(dm_list,"3dxy",1)
tp3dxy2=tpi(dm_list,"3dxy",2)
tz3dxz1=tz(dm_list,"3dxz",1)
tz3dxz2=tz(dm_list,"3dxz",2)

tsigsig=too(dm_list,"sig","sig")
tpipi=too(dm_list,"pi","pi")
tzz=too(dm_list,"z","z")
tss=too(dm_list,"s","s")
tsigpi=too(dm_list,"sig","pi")

data=np.concatenate(((e_list-e_list[0])[:,np.newaxis]*27.2114,nsum,
  ts4s1[:,np.newaxis],ts4s2[:,np.newaxis],
  ts3dz21[:,np.newaxis],ts3dz22[:,np.newaxis],
  ts3d1[:,np.newaxis],ts3d2[:,np.newaxis],
  
  tsig4s1[:,np.newaxis],tsig4s2[:,np.newaxis],
  tsig3dz21[:,np.newaxis],tsig3dz22[:,np.newaxis],
  tsig3d1[:,np.newaxis],tsig3d2[:,np.newaxis],
  
  tp3dxy1[:,np.newaxis],tp3dxy2[:,np.newaxis],
  tz3dxz1[:,np.newaxis],tz3dxz2[:,np.newaxis],
  
  tsigsig[:,np.newaxis],tpipi[:,np.newaxis],
  tss[:,np.newaxis],tzz[:,np.newaxis],tsigpi[:,np.newaxis]),
  axis=1)

labels=np.array(["E","5s","2s","2psg","2ppi","2pz","4s1","3dxy1",
        "3dyz1","3dz21","3dxz1","3d1","4s2","3dxy2",
        "3dyz2","3dz22","3dxz2","3d2",
        "ts4s1","ts4s2","ts3dz21","ts3dz22","ts3d1","ts3d2",
        "tsig4s1","tsig4s2","tsig3dz21","tsig3dz22","tsig3d1","tsig3d2",
        "tp3dxy1","tp3dxy2","tz3dxz1","tz3dxz2",
        "tsigsig","tpipi","tss","tzz","tsigpi"])
df=pd.DataFrame(data=data,columns=labels)
y=df["E"]
X=df.drop(["E"],axis=1)
labels=np.array(list(X))

'''
#Correlation matrix
plt.matshow(X.corr(),vmin=-1,vmax=1,cmap=plt.cm.bwr)
plt.xticks(np.arange(len(labels)),labels,rotation=90)
plt.yticks(np.arange(len(labels)),labels)
plt.colorbar()
plt.show()

#Block correlation matrix
ordering = find_connect.recursive_order(X.corr().values,[1e-10,1e-4,0.2,0.3,0.4,0.5,0.6,0.8,0.9,1])
rearrange = X.corr().values[ordering][:,ordering]
plt.matshow(rearrange,vmin=-1,vmax=1,cmap=plt.cm.bwr)
plt.xticks(np.arange(len(labels)),labels,rotation=90)
plt.yticks(np.arange(len(labels)),labels)
plt.colorbar()
plt.show()
exit(0)
'''

#Weighted multilinear regression  (https://www.statsmodels.org/dev/examples/notebooks/generated/wls.html)
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
wls = sm.WLS(y, X, weights=np.exp(-1*y))
res_wls = wls.fit()
#print(res_wls.summary())
prstd, iv_l, iv_u = wls_prediction_std(res_wls)
plt.plot(y,y,'k--')
plt.plot(y, res_wls.fittedvalues, 'bo')
plt.plot(y, iv_u, 'g.', label="WLS")
plt.plot(y, iv_l, 'g.')
plt.legend(loc="best")
plt.show()

exit(0)

'''
reg = LinearRegression().fit(X, y)
print(reg.coef_)
print(reg.intercept_)
print(reg.score(X,y))
plt.ylabel("E - E(CHK), eV")
plt.xlabel("E_pred")
plt.title("R^2="+str(reg.score(X,y)))
plt.plot(reg.predict(X),y,'o')
plt.plot(y,y,'-')
plt.show()
'''

#PCA
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X)

plt.plot(pca.explained_variance_ratio_,'s-')
plt.ylabel("Explained variance ratio")
plt.xlabel("PCA component")
plt.show()

plt.matshow(pca.components_,vmin=-1,vmax=1,cmap=plt.cm.bwr)
plt.xticks(np.arange(pca.components_.shape[0]),labels,rotation=90)
plt.yticks(np.arange(pca.components_.shape[0]))
plt.colorbar()
plt.show()

#General feature selection 
#Variance Thresholding
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold
cscores=[]
cscores_err=[]
scores=[]
conds=[]
nparms=[]
for th in np.linspace(0,0.16,100):
  sel=VarianceThreshold(threshold=th)
  Xr=sel.fit_transform(X)
  reg = LinearRegression().fit(Xr, y)
  nparms.append(reg.coef_.shape[0]+1)
  scores.append(reg.score(Xr,y))
  conds.append(np.linalg.cond(Xr))
  
  tmp=cross_val_score(reg,Xr,y,cv=5) 
  cscores.append(tmp.mean())
  cscores_err.append(tmp.std()*2)
plt.subplot(331)
plt.plot(nparms,scores,'o')
plt.subplot(332)
plt.errorbar(nparms,cscores,yerr=cscores_err,fmt='gs')
plt.subplot(333)
plt.plot(nparms,np.log(conds),'r*')

from sklearn.linear_model import OrthogonalMatchingPursuit
cscores=[]
cscores_err=[]
scores=[]
conds=[]
nparms=[]
for i in range(1,len(labels)):
  omp = OrthogonalMatchingPursuit(n_nonzero_coefs=i,fit_intercept=True)
  omp.fit(X, y)
  nparms.append(sum(np.abs(omp.coef_)>0)+1)
  scores.append(omp.score(X,y))
  Xr=X.drop(labels[omp.coef_==0],axis=1)
  conds.append(np.linalg.cond(Xr))
  
  tmp=cross_val_score(omp,X,y,cv=5) 
  cscores.append(tmp.mean())
  cscores_err.append(tmp.std()*2)
plt.subplot(334)
plt.plot(nparms,scores,'o')
plt.subplot(335)
plt.errorbar(nparms,cscores,yerr=cscores_err,fmt='gs')
plt.subplot(336)
plt.plot(nparms,np.log(conds),'r*')

from sklearn.linear_model import OrthogonalMatchingPursuitCV
cscores=[]
cscores_err=[]
scores=[]
conds=[]
nparms=[]
for i in range(1,len(labels)):
  omp = OrthogonalMatchingPursuitCV(cv=5,max_iter=i,fit_intercept=True)
  omp.fit(X, y)
  nparms.append(sum(np.abs(omp.coef_)>0)+1)
  scores.append(omp.score(X,y))
  Xr=X.drop(labels[omp.coef_==0],axis=1)
  conds.append(np.linalg.cond(Xr))
  
  tmp=cross_val_score(omp,X,y,cv=5) 
  cscores.append(tmp.mean())
  cscores_err.append(tmp.std()*2)
plt.subplot(337)
plt.plot(nparms,scores,'o')
plt.subplot(338)
plt.errorbar(nparms,cscores,yerr=cscores_err,fmt='gs')
plt.subplot(339)
plt.plot(nparms,np.log(conds),'r*')
plt.show()

#Restricted feature selection
from sklearn.linear_model import OrthogonalMatchingPursuit
import matplotlib as mpl

n_lines=10
ax = plt.axes()
ax.set_color_cycle([plt.cm.Blues(i) for i in np.linspace(0, 1, n_lines)])
for i in range(12,22):
  omp = OrthogonalMatchingPursuit(n_nonzero_coefs=i,fit_intercept=True)
  omp.fit(X,y)
  plt.plot(np.arange(len(omp.coef_)+1),[omp.intercept_]+list(omp.coef_),'o-',label=i+1)
plt.legend(loc=2)
labels[0]="E0"
plt.axhline(0,color='k',linestyle='--')
plt.xticks(np.arange(len(list(X))+1),["E0"]+list(X),rotation=90)
plt.show()