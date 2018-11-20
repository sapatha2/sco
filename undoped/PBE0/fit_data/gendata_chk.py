#Generate excitations, energies and rdms for excitations built on CHK state
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from crystal2pyscf import crystal2pyscf_cell
from pyscf2qwalk import print_qwalk_mol
from methods import calcIAO, rdmIAO, hIAO, genex, data_from_ex, getn, nsum 
from methods import tsig3d1, tsig3d2, tsig4s1, tsig4s2, tsig3dz1, tsig3dz2, ts3d1, ts3d2, ts4s1, ts4s2, ts3dz1, ts3dz2
from methods import tp3dxy1, tp3dxy2, tz3dyz1, tz3dyz2
from basis import basis, minbasis, basis_order
import seaborn as sns
sns.set_style("white")
from sklearn.linear_model import LinearRegression

###########################################################################################
#Build IAO 
direc="../CHK"
occ=[i for i in range(72)]
cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=direc+"/GRED.DAT",kred=direc+"/KRED.DAT",totspin=0)
a=calcIAO(cell,mf,minbasis,occ)

#Build excitations
occ=np.arange(24,66)
virt=np.arange(66,72)
ex='singles'
ex_list=genex(mf.mo_occ,[occ,occ],[virt,virt],ex=ex)
e_list,rdm_list=data_from_ex(mf,a,ex_list)
print(ex_list.shape,e_list.shape,rdm_list.shape)

#Analysis
n=getn(rdm_list)
nsum=nsum(n)
tsig3d1=tsig3d1(rdm_list)
tsig3d2=tsig3d2(rdm_list)
tsig4s1=tsig4s1(rdm_list)
tsig4s2=tsig4s2(rdm_list)
tsig3dz1=tsig3dz1(rdm_list)
tsig3dz2=tsig3dz2(rdm_list)
ts3d1=ts3d1(rdm_list)
ts3d2=ts3d2(rdm_list)
ts4s1=ts4s1(rdm_list)
ts4s2=ts4s2(rdm_list)
ts3dz1=ts3dz1(rdm_list)
ts3dz2=ts3dz2(rdm_list)
tp3dxy1=tp3dxy1(rdm_list)
tp3dxy2=tp3dxy2(rdm_list)
tz3dyz1=tz3dyz1(rdm_list)
tz3dyz2=tz3dyz2(rdm_list)
data=np.concatenate(((e_list-e_list[0])[:,np.newaxis]*27.2114,nsum,
  tsig3d1[:,np.newaxis],tsig3d2[:,np.newaxis],
  tsig4s1[:,np.newaxis],tsig4s2[:,np.newaxis],
  tsig3dz1[:,np.newaxis],tsig3dz2[:,np.newaxis],
  ts3d1[:,np.newaxis],ts3d2[:,np.newaxis],
  ts4s1[:,np.newaxis],ts4s2[:,np.newaxis],
  ts3dz1[:,np.newaxis],ts3dz2[:,np.newaxis],
  tp3dxy1[:,np.newaxis],tp3dxy2[:,np.newaxis],
  tz3dyz1[:,np.newaxis],tz3dyz2[:,np.newaxis]),
  axis=1)
labels=["E","5s","2s","2psg","2ppi","2pz","4s1","3dxy1",
        "3dyz1","3dz21","3dxz1","3d1","4s2","3dxy2",
        "3dyz2","3dz22","3dxz2","3d2",
        "tsig3d1","tsig3d2","tsig4s1","tsig4s2","tsig3dz1","tsig3dz2",
        "ts3d1","ts3d2","ts4s1","ts4s2","ts3dz1","ts3dz2",
        "tp3dxy1","tp3dxy2","tz3dyz1","tz3dyz2"]
df=pd.DataFrame(data=data,columns=labels)

#Correlation matrix
'''
fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(df.corr(),vmin=-1.0,vmax=1.0,cmap=plt.cm.bwr)
plt.xticks(np.arange(18),labels,rotation=90)
plt.yticks(np.arange(18),labels)
fig.colorbar(cax)
plt.show()
'''

#Pairplot
'''
sns.pairplot(df[["E","2ppi","3dxy1","3dxy2","2pz","3dxz1","3dxz2","3dz21","3dz22","3d1"]],diag_kind='kde') #Classifiers
sns.pairplot(df[["E","2s","2psg","4s1","4s2","3d2"]],diag_kind='kde') #Regressors
plt.show()
'''
#sns.pairplot(df[["E","2s","2psg","4s1","4s2","3d2","tsig"]],diag_kind='kde') #Regressors
#plt.show()

#Multilinear regression 
y=df["E"]
X=df.drop("E",axis=1) #Use all! Big collinearity among occupations, might need to do PCA there
reg = LinearRegression().fit(X, y)
print(reg.coef_)
print(reg.intercept_)
print(reg.score(X,y))
plt.plot(reg.predict(X),y,'o')
plt.plot(y,y,'-')
plt.show()

#PCA
'''
from sklearn.decomposition import PCA
X=df.drop("E",axis=1)
pca = PCA()
pca.fit(X)

print(pca.explained_variance_ratio_)
plt.matshow(pca.components_,vmin=-1,vmax=1,cmap=plt.cm.bwr)
plt.xticks(np.arange(pca.components_.shape[0]),labels[1:],rotation=90)
plt.yticks(np.arange(pca.components_.shape[0]))
plt.colorbar()
plt.show()
'''
