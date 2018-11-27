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

df=pd.read_pickle("dft_model.pickle")
exit(0)

#Collect selected parameters
psums=[]
parameters=[]
labels=[]
for i in range(len(full_labels)):
  sp=full_labels[i].split("-")
  if(("3s" in full_labels[i]) or ("3p" in full_labels[i])): pass #Remove 3s, 3p, 2s
  elif(("5s" in full_labels[i])): pass #Remove 5s
  else: 
    '''
    #Number only
    #if((sp[0]==sp[1])):
    
    #Number, Cu-O hopping only 
    if((sp[0]==sp[1]) or ((sp[0][0]=="c") and (sp[1][0]=="o")) or
    ((sp[1][0]=="c") and (sp[0][0]=="o"))):
      psums.append(full_psums[i])
      parameters.append(full_parameters[i])
      labels.append(full_labels[i])
    
    #2s, 2psig, 4s, 3dx2y2 space only, Cu-O hopping only 
    if((("2s" in sp[0]) or ("2psig" in sp[0]) or ("4s" in sp[0]) or ("3dx2y2" in sp[0]))
    and (("2s" in sp[1]) or ("2psig" in sp[1]) or ("4s" in sp[1]) or ("3dx2y2" in sp[1]))):
      if((sp[0][0]==sp[1][0]) and (sp[0]!=sp[1])): pass
      else:
    '''
    psums.append(full_psums[i])
    parameters.append(full_parameters[i])
    labels.append(full_labels[i])
psums=np.array(psums)
parameters=np.array(parameters)
labels=np.array(labels)

print("Full size: ",full_psums.shape,full_parameters.shape,len(full_labels))
print("Without 3s, 3p: ",psums.shape,parameters.shape,len(labels))

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
var=[]
for i in range(psums.shape[0]):
  var.append(np.var(psums[i,:]))
ind=np.argsort(var)
plt.plot(ind,np.array(var)[ind],'bo')
plt.yticks(var,labels)
plt.show()

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


#OMP
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import OrthogonalMatchingPursuit
cscores=[]
cscores_err=[]
scores=[]
conds=[]
nparms=[]
#Full OMP
n_lines=X.shape[1]
ax = plt.axes()
ax.set_color_cycle([plt.cm.Blues(i) for i in np.linspace(0, 1, n_lines)])
for i in range(1,X.shape[1]+1):
  print("n_nonzero_coefs="+str(i))
  omp = OrthogonalMatchingPursuit(n_nonzero_coefs=i,fit_intercept=True)
  omp.fit(X,y)
  nparms.append(i)
  scores.append(omp.score(X,y))
  tmp=cross_val_score(omp,X,y,cv=5)
  cscores.append(tmp.mean())
  cscores_err.append(tmp.std()*2)
  print("R2CV: ",tmp.mean(),"(",tmp.std()*2,")")
  ind=np.abs(omp.coef_)>0
  Xr=X[:,ind]
  conds.append(np.linalg.cond(Xr))
  #Low condition number, high R2
  if((conds[-1] < 100000) and (tmp.mean()>0.90)): plt.plot(np.arange(len(omp.coef_)),omp.coef_,'o-')
plt.plot(np.arange(len(omp.coef_)),parameters*27.2114,'k*-')
plt.axhline(0,color='k',linestyle='--')
plt.xticks(np.arange(len(labels)),labels,rotation=60)
plt.show()

plt.subplot(131)
plt.plot(nparms,scores,'o')
plt.subplot(132)
plt.errorbar(nparms,cscores,yerr=cscores_err,fmt='gs')
plt.subplot(133)
plt.plot(nparms,conds,'r*')
plt.show()

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
