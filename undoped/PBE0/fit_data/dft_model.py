#Generate excitations, energies and rdms for excitations built on CHK state
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from crystal2pyscf import crystal2pyscf_cell
from methods import calcIAO, rdmIAO, hIAO, group, genex, data_from_ex, gen_sumsingles
from methods import ts, tsig, tpi, tz, too
from basis import basis, minbasis, basis_order
import seaborn as sns
sns.set_style("white")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import find_connect

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
#Build H1 on CHK state
h=hIAO(mf,a)        
df,unique_vals=group(3,0.0,h,labels,out=1)
print("Finished H1 build")

#Generate excitations on CHK
'''
ex='singles'
occ=np.arange(16,66) #2s - homo
virt=np.arange(66,72) #homo - Sr occup
ex_list,q,r=genex(mf.mo_occ,[occ,occ],[virt,virt],ex=ex)
e_list,dm_list=data_from_ex(mf,a,ex_list)
'''

ex='sum_singles'
Ndet=10
c=0.1
N=500
occ=np.arange(16,66) #2s - homo
virt=np.arange(66,72) #homo - Sr occup
ex_list,q,r,spin=genex(mf.mo_occ,[occ,occ],[virt,virt],ex=ex)
e_list,dm_list=data_from_ex(mf,a,ex_list)
e_list,dm_list=gen_sumsingles(e_list,dm_list,ex_list,c=c,Ndet=Ndet,N=N,q=q,r=r,spin=spin,mf=mf,a=a)
print("Finished excitation build")

#Collect unique parameters and parameter sums for those 
parameters=[]
psums=[]
labels=[]
for u in unique_vals[::-1]:
  #dfs=df[df['round']==u][["s","i1","i2","e","c1","c2"]].values.T
  dfs=df[df['round']==u][["s","i1","i2","e"]].values.T
  s=dfs[0].astype(int)
  i1=dfs[1].astype(int)
  i2=dfs[2].astype(int)
  sign=np.sign(dfs[3])
  #labels.append(dfs[4][0]+"-"+dfs[5][0]+"_"+str(dfs[0][0]))
  psums.append(np.dot(dm_list[:,s,i1,i2],sign/sign[0]))
  parameters.append(dfs[3][0])
psums=np.array(psums)
parameters=np.array(parameters)
print("Finished parameters build")

#Plot
pred=np.einsum('ji,j->i',psums,parameters)
print(len(parameters),r2_score(pred-pred[0],e_list-e_list[0]))
plt.plot(e_list-e_list[0],pred-pred[0],'o')
plt.plot(e_list-e_list[0],e_list-e_list[0],'-')
plt.ylabel("Predicted energy")
plt.xlabel("PBE0 eigenvalue differences")
plt.show()
exit(0)

#OMP
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import OrthogonalMatchingPursuit
cscores=[]
cscores_err=[]
scores=[]
conds=[]
nparms=[]
X=psums.T 
#X=np.roll(X,-4,axis=1) #Move 5s to the end
y=e_list[:,np.newaxis]

'''
for i in range(1,X.shape[1]):
  omp = OrthogonalMatchingPursuit(n_nonzero_coefs=i,fit_intercept=True)
  omp.fit(X,y)
  nparms.append(i)
  scores.append(omp.score(X,y))
  #Xr=X.drop(labels[omp.coef_==0],axis=1)
  #conds.append(np.linalg.cond(Xr))

  tmp=cross_val_score(omp,X,y,cv=5)
  cscores.append(tmp.mean())
  cscores_err.append(tmp.std()*2)

plt.subplot(131)
plt.plot(nparms,scores,'o')
plt.subplot(132)
plt.errorbar(nparms,cscores,yerr=cscores_err,fmt='gs')
#plt.subplot(133)
#plt.plot(nparms,np.log(conds),'r*')
plt.show()
'''

#Restricted OMP
#n_lines=40
#ax = plt.axes()
#ax.set_color_cycle([plt.cm.Blues(i) for i in np.linspace(0, 1, n_lines)])
for i in range(10,59):
  omp = OrthogonalMatchingPursuit(n_nonzero_coefs=i)
  omp.fit(X,y)
  print(omp.score(X,y))
  plt.plot(np.arange(len(omp.coef_)),-np.sort(-np.abs(list(omp.coef_))),'o-')
  #plt.plot(e_list,omp.predict(X),'.')
  #plt.show()
  #exit(0)
plt.axhline(0,color='k',linestyle='--')
plt.plot(np.arange(len(omp.coef_)),-np.sort(-np.abs(parameters)),'k*-')
#plt.xticks(np.arange(len(labels)),labels[np.argsort(np.abs(omp.coef_))],rotation=60)
plt.show()
