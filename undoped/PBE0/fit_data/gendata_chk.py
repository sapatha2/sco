#Generate excitations, energies and rdms for excitations built on CHK state
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from crystal2pyscf import crystal2pyscf_cell
from pyscf2qwalk import print_qwalk_mol
from methods import calcIAO, rdmIAO, hIAO, genex, data_from_ex, getn, nsum 
from methods import ts, tsig, tpi, tz
from basis import basis, minbasis, basis_order
import seaborn as sns
sns.set_style("white")
from sklearn.linear_model import LinearRegression

###########################################################################################
#Build IAO 
direc="../CHK"
#occ=[i for i in range(72)]
occ=[i for i in range(69)]
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
occ=np.arange(24,66)
virt=np.arange(66,72)
ex='singles'
ex_list=genex(mf.mo_occ,[occ,occ],[virt,virt],ex=ex)
e_list,dm_list=data_from_ex(mf,a,ex_list)
print(ex_list.shape,e_list.shape,dm_list.shape)

'''
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

data=np.concatenate(((e_list-e_list[0])[:,np.newaxis]*27.2114,nsum,
  ts4s1[:,np.newaxis],ts4s2[:,np.newaxis],
  ts3dz21[:,np.newaxis],ts3dz22[:,np.newaxis],
  ts3d1[:,np.newaxis],ts3d2[:,np.newaxis],
  
  tsig4s1[:,np.newaxis],tsig4s2[:,np.newaxis],
  tsig3dz21[:,np.newaxis],tsig3dz22[:,np.newaxis],
  tsig3d1[:,np.newaxis],tsig3d2[:,np.newaxis],
  
  tp3dxy1[:,np.newaxis],tp3dxy2[:,np.newaxis],
  tz3dxz1[:,np.newaxis],tz3dxz2[:,np.newaxis]),
  axis=1)

labels=["E","5s","2s","2psg","2ppi","2pz","4s1","3dxy1",
        "3dyz1","3dz21","3dxz1","3d1","4s2","3dxy2",
        "3dyz2","3dz22","3dxz2","3d2",
        "ts4s1","ts4s2","ts3dz21","ts3dz22","ts3d1","ts3d2",
        "tsig4s1","tsig4s2","tsig3dz21","tsig3dz22","tsig3d1","tsig3d2",
        "tp3dxy1","tp3dxy2","tz3dxz1","tz3dxz2"]
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
#X=df.drop("E","5s",axis=1) #Use all! Big collinearity among occupations, might need to do PCA there
X=df.drop("E",axis=1) #Use all! Big collinearity among occupations, might need to do PCA there
X=df.drop("5s",axis=1) #Use all! Big collinearity among occupations, might need to do PCA there
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

#PCA
from sklearn.decomposition import PCA
X=df.drop("E",axis=1)
pca = PCA()
pca.fit(X)

plt.plot(pca.explained_variance_ratio_,'s-')
plt.ylabel("Explained variance ratio")
plt.xlabel("PCA component")
plt.show()

plt.matshow(pca.components_,vmin=-1,vmax=1,cmap=plt.cm.bwr)
plt.xticks(np.arange(pca.components_.shape[0]),labels[1:],rotation=90)
plt.yticks(np.arange(pca.components_.shape[0]))
plt.colorbar()
plt.show()
