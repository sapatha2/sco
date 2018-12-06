#Generate excitations, energies and rdms for excitations built on CHK state
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from crystal2pyscf import crystal2pyscf_cell
from methods import genex
from basis import basis, minbasis, basis_order
import seaborn as sns
sns.set_style("white")
from pyscf import lo
from functools import reduce 
from pyscf2qwalk import print_qwalk_pbc

def calcIAO(cell,mf,basis,occ):
  ''' 
  input: 
  cell and scf (PBC SCF) objects from pyscf and basis
  to calculate IAOs on
  occ is MOs which IAOs should span
  output:
  Calculates 1RDM on orthog atomic orbitals, orthogonalized
  using Lowdin S^1/2
  Returns coefficient matrix for IAOs 
  '''
  s=mf.get_ovlp()[0]

  mo_occ = mf.mo_coeff[0][0][:,occ[0]]
  mo_occ2 = mf.mo_coeff[1][0][:,occ[1]]
  mo_occ=np.concatenate((mo_occ,mo_occ2),axis=1)
  a = lo.iao.iao(cell, mo_occ, minao=basis)
  a = lo.vec_lowdin(a, s)
 
  return a

def rdmIAO(mf,a,occ):
  '''
  input:
  mf object for calculation
  output:
  1rdm for spin up and down
  '''
  s=mf.get_ovlp()[0]
  ind = np.nonzero(mf.mo_occ[0][0][occ[0]]*occ[0])
  mo_occ = mf.mo_coeff[0][0][:,occ[0][ind]]
  mo_occ = reduce(np.dot, (a.T, s, mo_occ))
  dm_u = np.dot(mo_occ, mo_occ.T)
  ind = np.nonzero(mf.mo_occ[1][0][occ[1]]*occ[1])
  mo_occ = mf.mo_coeff[1][0][:,occ[1][ind]]
  mo_occ = reduce(np.dot, (a.T, s, mo_occ))
  dm_d = np.dot(mo_occ, mo_occ.T)
  return np.array([dm_u,dm_d])

#Labels
def makelabels():
  siglist=["2px_1","2px_2","2px_3","2px_4","2py_5","2py_6","2py_7","2py_8"]
  pilist= ["2py_1","2py_2","2py_3","2py_4","2px_5","2px_6","2px_7","2px_8"]
  culabels=["3s","4s","3px","3py","3pz",'3dxy','3dyz','3dz2','3dxz','3dx2y2']
  oxlabels=["2s","2px","2py","2pz"]
  orig_labels=[]
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
  return labels

###########################################################################################
#Build IAO 
direc="../FLP_ns"
act_mo=[np.arange(67,73)-1,np.arange(66,73)-1]
cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=direc+"/GRED.DAT",kred=direc+"/KRED.DAT",totspin=1)
a=calcIAO(cell,mf,minbasis,act_mo)

#Spin state to work with 
direc="../COL_ns"
act_mo=[np.arange(67,73)-1,np.arange(66,73)-1]
cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=direc+"/GRED.DAT",kred=direc+"/KRED.DAT",totspin=1)
dm=rdmIAO(mf,a,act_mo)

###########################################################################################
#Testing
#IAO information
print(direc)
print("Nelectron, MO RDM: ",sum(mf.mo_occ[0][0][act_mo[0]]),sum(mf.mo_occ[1][0][act_mo[1]]))
print("Nelectron, IAO RDM: ",np.trace(dm[0]),np.trace(dm[1]))
print("Finished IAO build")
'''
IAO basis: FLP1
Tr FLP1    0.9980959566006015 0.9970811630848089
Tr COL1    0.9979964219133656 0.9936417711415055
Tr COL2    0.9976157716115474 0.9987198122216554   
Tr FM   -  HAS SOME BIG ISSUES, NEED TO THINK
'''

#Plot IAO
'''
for i in range(a.shape[1]):
  mf.mo_coeff[0][0][:,i]=a[:,i]
print_qwalk_pbc(cell,mf,basename='FLP1_iao')
'''

#Occupations 
'''
labels=makelabels()
plt.plot(np.diag(dm[0]),'*')
plt.plot(np.diag(dm[1]),'o')
plt.xticks(np.arange(len(labels)),labels,rotation=90)
plt.show()
'''

#MO to IAO matrix
'''
s=mf.get_ovlp()[0]
labels=makelabels()
M0=reduce(np.dot,(a.T,s,mf.mo_coeff[0][0])).T
plt.matshow(M0[act_mo[0],:],vmin=-1,vmax=1,cmap=plt.cm.bwr)
plt.xticks(np.arange(len(labels)),labels,rotation=90)
plt.show()
M0=reduce(np.dot,(a.T,s,mf.mo_coeff[1][0])).T
plt.matshow(M0[act_mo[1],:],vmin=-1,vmax=1,cmap=plt.cm.bwr)
plt.xticks(np.arange(len(labels)),labels,rotation=90)
plt.show()
'''

#Build excitations on base state
ncore=[66,65]
nact=[1,1]
N=50
Ndet=2
c=0.0
detgen='sd'
e_list,dm_list,iao_dm_list=genex(mf,a,ncore,nact,act_mo,N,Ndet,detgen,c)
tr=np.einsum('isjj->is',iao_dm_list)

'''
plt.plot(tr[:,0],'bo')
plt.plot(tr[:,1],'go')
plt.xlabel("Excitation")
plt.ylabel("Trace")
plt.show()

labels=makelabels()
for i in range(N):
  plt.plot(np.diag(iao_dm_list[i,0,:,:]),'bo')
  plt.plot(np.diag(iao_dm_list[i,1,:,:]),'go')
plt.xticks(np.arange(len(labels)),labels,rotation=90)
plt.ylabel("Occupation")
plt.show()
'''
