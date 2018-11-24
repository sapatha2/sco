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

###########################################################################################
#Build IAO 
direc="../CHK"
occ=[i for i in range(72)]
cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=direc+"/GRED.DAT",kred=direc+"/KRED.DAT",totspin=0)
a=calcIAO(cell,mf,minbasis,occ)
print("Finished IAO build")

#Labels
siglist=["2px_1","2px_2","2px_3","2px_4","2py_5","2py_6","2py_7","2py_8"]
pilist= ["2py_1","2py_2","2py_3","2py_4","2px_5","2px_6","2px_7","2px_8"]
culabels=["3s","4s","3px","3py","3pz",'3dxy','3dyz','3dz2','3dxz','3dx2y2']
oxlabels=["2s","2px","2py","2pz"]
srlabels=["5s"]
orig_labels=[]
for i in range(4):
  orig_labels+=[x+"_"+str(i+1) for x in srlabels]
for i in range(4):
  orig_labels+=[x+"_"+str(i+1) for x in culabels]
for i in range(8):
  orig_labels+=[x+"_"+str(i+1) for x in oxlabels]
labels=orig_labels

my_labels=[]
for i in range(len(labels)):
  if(labels[i] in siglist): my_labels.append('2psig_'+labels[i].split("_")[1])
  elif(labels[i] in pilist): my_labels.append('2ppi_'+labels[i].split("_")[1])
  else: my_labels.append(labels[i])
labels=my_labels[:]

###########################################################################################
#DATA GENERATION

#Build H1 on CHK state
h=hIAO(mf,a)        
df,unique_vals=group(3,0.0,h,labels,out=1)
print("Finished H1 build")

#Generate excitations on CHK
Ndet=10
c=0.90
N=500
detgen='a' #all
#detgen='s' #singles

ncore=16 #Core orbitals
act=[np.arange(16,72),np.arange(16,72)] #List of active orbitals, spin separated
nact=[50,50] #Number of active electrons
e_list,mo_dm_list,dm_list=genex(mf,a,ncore,act,nact,N,Ndet,detgen,c)
print("Finished excitation build")

#Collect full parameters
full_parameters=[]
full_psums=[]
full_labels=[]
for u in unique_vals[::-1]:
  dfs=df[df['round']==u][["s","i1","i2","e"]].values.T
  s=dfs[0].astype(int)
  i1=dfs[1].astype(int)
  i2=dfs[2].astype(int)
  sign=np.sign(dfs[3])
  full_psums.append(np.dot(dm_list[:,s,i1,i2],sign/sign[0]))
  full_parameters.append(dfs[3][0])
  ldf=df[df['round']==u][["s","c1","c2"]].values.T
  full_labels.append(ldf[1][0]+"-"+ldf[2][0]+"_"+str(ldf[0][0]))
full_psums=np.array(full_psums)
full_parameters=np.array(full_parameters)
print("Finished parameters build")

#Collect selected parameters
psums=[]
parameters=[]
labels=[]
for i in range(len(full_labels)):
  if(("3s" in full_labels[i]) or ("3p" in full_labels[i])
  or ("5s" in full_labels[i])): pass #Remove 3p, 3s, 5s
  else: 
    psums.append(full_psums[i])
    parameters.append(full_parameters[i])
    labels.append(full_labels[i])
psums=np.array(psums)
parameters=np.array(parameters)

print(full_psums.shape,full_parameters.shape,len(full_labels))
print(psums.shape,parameters.shape,len(labels))

###########################################################################################
#ANALYSIS
#Plot fit from rotated H1
#Full parameters
full_pred=np.einsum('ji,j->i',full_psums,full_parameters)
a=(full_pred-full_pred[0])*27.2114
b=(e_list-e_list[0])*27.2114
print(len(full_parameters),r2_score(a,b))
plt.plot(b,a,'bo')

'''
#Select parameters
pred=np.einsum('ji,j->i',psums,parameters)
a=(pred-pred[0])*27.2114
b=(e_list-e_list[0])*27.2114
print(len(parameters),r2_score(a,b))
plt.plot(b,a,'go')
'''

plt.plot(b,b,'-')
plt.ylabel("Predicted energy")
plt.xlabel("PBE0 eigenvalue differences")
plt.show()

exit(0)

'''
X=psums.T
y=e_list[:,np.newaxis]*27.2114
data=np.concatenate((y,X),axis=1)
df=pd.DataFrame(data)
plt.matshow(df.corr(),vmin=-1,vmax=1,cmap=plt.cm.bwr)
plt.colorbar()
plt.xticks(np.arange(len(labels)+1),["E"]+labels,rotation=90)
plt.xticks(np.arange(len(labels)+1),["E"]+labels)
plt.show()
'''

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import OrthogonalMatchingPursuit
cscores=[]
cscores_err=[]
scores=[]
conds=[]
nparms=[]

X=psums.T
y=e_list[:,np.newaxis]*27.2114

#Full OMP
for i in range(1,X.shape[1]):
  omp = OrthogonalMatchingPursuit(n_nonzero_coefs=i,fit_intercept=True)
  omp.fit(X,y)
  nparms.append(i)
  scores.append(omp.score(X,y))
  tmp=cross_val_score(omp,X,y,cv=5)
  cscores.append(tmp.mean())
  cscores_err.append(tmp.std()*2)
  ind=np.abs(omp.coef_)>0
  Xr=X[:,ind]
  conds.append(np.linalg.cond(Xr))
plt.subplot(131)
plt.plot(nparms,scores,'o')
plt.subplot(132)
plt.errorbar(nparms,cscores,yerr=cscores_err,fmt='gs')
plt.subplot(133)
plt.plot(nparms,conds,'r*')
plt.show()

#Restricted OMP
'''
plt.subplot(211)
n_lines=X.shape[1]
ax = plt.axes()
ax.set_color_cycle([plt.cm.bwr(i) for i in np.linspace(0, 1, n_lines)])
for i in range(1,X.shape[1]-1):
  omp = OrthogonalMatchingPursuit(n_nonzero_coefs=i,fit_intercept=True)
  omp.fit(X,y)
  print(omp.score(X,y))
  plt.plot(np.arange(len(omp.coef_)),omp.coef_,'o-')
plt.plot(np.arange(len(omp.coef_)),parameters*27.2114,'k*-')
plt.axhline(0,color='k',linestyle='--')
plt.xticks(np.arange(len(labels)),labels,rotation=60)
plt.show()
'''
