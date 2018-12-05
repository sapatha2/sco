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
from functools import reduce 

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
  orig_labels+=["sr"+x+"_"+str(i+1) for x in srlabels]
for i in range(4):
  orig_labels+=["cu"+x+"_"+str(i+1) for x in culabels]
for i in range(8):
  orig_labels+=["o"+x+"_"+str(i+1) for x in oxlabels]
labels=orig_labels

my_labels=[]
for i in range(len(labels)):
  if(labels[i][1:] in siglist): my_labels.append('o2psig_'+labels[i].split("_")[1])
  elif(labels[i][1:] in pilist): my_labels.append('o2ppi_'+labels[i].split("_")[1])
  else: my_labels.append(labels[i])
labels=my_labels[:]

###########################################################################################
#DATA GENERATION
#Build H1 on CHK state
h=hIAO(mf,a)
df,unique_vals=group(3,0.0,h,labels,out=1)
print("Finished H1 build")
print(len(unique_vals))

#Generate excitations on CHK
Ndet=5
c=0.7
N=500
detgen='s'
ncore=65
act=[np.arange(66,72),np.arange(66,72)]
nact=[1,1]
#ncore=16 #Core orbitals (no 3s, 3p, 2s)
#act=[np.arange(16,72),np.arange(16,72)] #List of active orbitals, spin separated
#nact=[50,50] #Number of active electrons

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
  full_labels.append(ldf[1][0]+"-"+ldf[2][0]+"-"+str(ldf[0][0]))
full_psums=np.array(full_psums)
full_parameters=np.array(full_parameters)
print("Finished parameters build")

X=full_psums.T
y=e_list[:,np.newaxis]
data=np.concatenate((y,X),axis=1)
df=pd.DataFrame(data,columns=["E"]+full_labels)
df.to_pickle("dft_model_iao3_"+detgen+"_"+str(ncore)+"_"+str(Ndet)+"_"+str(c)+".pickle")
