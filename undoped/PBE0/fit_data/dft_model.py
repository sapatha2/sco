#Generate excitations, energies and rdms for excitations built on CHK state
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from crystal2pyscf import crystal2pyscf_cell
from methods import calcIAO, rdmIAO, hIAO, group, genex, data_from_ex
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
ex='singles'
occ=np.arange(24,66)
virt=np.arange(66,72)
ex_list=genex(mf.mo_occ,[occ,occ],[virt,virt],ex=ex)
e_list,dm_list=data_from_ex(mf,a,ex_list)
print("Finished excitation build")

#Collect unique parameters and parameter sums for those 
parameters=[]
psums=[]
for u in unique_vals[::-1]:
  dfs=df[df['round']==u][["s","i1","i2","e"]].values.T
  s=dfs[0].astype(int)
  i1=dfs[1].astype(int)
  i2=dfs[2].astype(int)
  sign=np.sign(dfs[3])
  psums.append(np.dot(dm_list[:,s,i1,i2],sign/sign[0]))
  parameters.append(dfs[3][0])
psums=np.array(psums)
parameters=np.array(parameters)
print("Finished parameters build")

#Plot
pred=np.einsum('ji,j->i',psums,parameters)
pred-=pred[0]
print(len(parameters),r2_score(pred,e_list))
plt.plot(e_list,pred,'o')
plt.plot(e_list,e_list,'--')
plt.ylabel("Predicted energy")
plt.xlabel("PBE0 eigenvalue differences")
plt.show()

