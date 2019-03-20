#BUILD IAOS AND COLLECT MOS
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from crystal2pyscf import crystal2pyscf_cell
from pyscf2qwalk import print_qwalk_pbc
from basis import basis, minbasis, basis_order
from pyscf import lo, mcscf, fci, scf, df
from copy import deepcopy
from functools import reduce 
from downfold_tools import gen_slater_tbdm,sum_onebody,sum_J,sum_U

###########################################################################################
#Load SCF results
d='../../../PBE0/CHK'
cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=d+"/GRED.DAT",kred=d+"/KRED.DAT",totspin=0)

#Load IAOs
a=np.load('../iao.pickle')
s=mf.get_ovlp()
M0=reduce(np.dot,(a.T, s, mf.mo_coeff[0][0])) 
M1=reduce(np.dot,(a.T, s, mf.mo_coeff[1][0])) 

###############################################################
#Method to get RDM
def calc_1rdm(w,ex):
  #RDM for each determinant
  mo_dm=np.einsum('ijk,kl->ijkl',ex,np.eye(ex.shape[2],ex.shape[2]))
  
  #Calculate RDM sum
  dl=np.zeros((mo_dm.shape[1],mo_dm.shape[2],mo_dm.shape[2]))
  Ndet=mo_dm.shape[0]
  for n in range(Ndet):
    dl+=mo_dm[n,:,:,:]*w[n]**2
    
    #Cross terms
    diag0=np.diag(mo_dm[n,0,:,:]-mo_dm[0,0,:,:])
    diag1=np.diag(mo_dm[n,1,:,:]-mo_dm[0,1,:,:])
    i0=np.where(diag0!=0)[0]
    i1=np.where(diag1!=0)[0]
    if(i0.shape[0]>0):
      dl[0,i0[0],i0[1]]+=w[0]*w[n]
      dl[0,i0[1],i0[0]]+=w[0]*w[n]
    if(i1.shape[0]>0):
      dl[1,i1[0],i1[1]]+=w[0]*w[n]
      dl[1,i1[1],i1[0]]+=w[0]*w[n]
  
  #Normalize
  dl/=np.dot(w,w)

  #Move to IAO basis
  R=np.einsum('ij,jk->ik',dl[0],M0.T)
  dm_u=np.einsum('ij,jk->ik',M0,R)
  R=np.einsum('ij,jk->ik',dl[1],M1.T)
  dm_d=np.einsum('ij,jk->ik',M1,R)
  return np.array([dm_u,dm_d])

#Method to get position in descriptor space
def get_r(obdm):
  orb1=np.array([14,14,14,14,24,24,24,24,34,34,34,34,44,44,44,44])-1
  orb2=np.array([46,63,54,67,50,67,58,63,54,71,46,75,58,75,50,71])-1
  sign=np.array([-1,1,1,-1]*4)
  sigT=sum_onebody(obdm,orb1,orb2)
  sigT=2*np.dot(sign,sigT)
  orb1=np.array([46,50,54,58,63,67,71,75])-1
  sigNps=np.sum(sum_onebody(obdm,orb1,orb1))
  orb1=np.array([14,24,34,44])-1
  sigNd=np.sum(sum_onebody(obdm,orb1,orb1))
  return [sigT,sigNps,sigNd]

###############################################################
#Parameters
T=1000 #MC steps
Nstate=3 #3 states
Nop=3    #3 operators
beta=5   #"Temperature"
d=1.0    #Move step size

#Define active space
act=[25,26,27,47,48,52,56]
virt=[67]*7
ex_list=[]
ex_list.append(mf.mo_occ[:,0,:])
for i in range(len(act)):
  t=deepcopy(mf.mo_occ[:,0,:])
  t[0][act[i]]=0
  t[0][virt[i]]=1
  assert(sum(t[0])==66)
  ex_list.append(t)
ex_list=np.array(ex_list)

#Initial coefficients
#w=np.random.normal(size=(Nstate,len(ex_list)))
w=np.zeros((Nstate,len(ex_list)))
w[:,0]=np.ones(Nstate)
r=np.zeros((Nstate,Nop))
rp=np.zeros((Nstate,Nop))
for t in range(T):
  #Propose a new position vector
  wp=w+d*np.random.normal(size=w.shape)

  #Get current and proposed position vectors
  for state in range(Nstate):
    obdm=calc_1rdm(w[state,:],ex_list)
    r[state]=get_r(obdm)
    obdmp=calc_1rdm(wp[state,:],ex_list)
    rp[state]=get_r(obdmp)
 
  #Calculate distances
  d0=0
  dp=0
  for state in range(Nstate):
    for statep in range(state+1,Nstate):
      d0+=np.dot(r[state]-r[statep],r[state]-r[statep])
      dp+=np.dot(rp[state]-rp[statep],rp[state]-rp[statep])

  #Calculate acceptance, move walker
  a=np.exp(-1*beta/dp)/np.exp(-1*beta/d0)
  acc=(a>np.random.uniform())
  print(d0,dp,a,acc)
  #if(t==5): exit(0)
  if(acc): 
    w=deepcopy(wp); r=deepcopy(rp)
    #print("Step: "+str(t)+", delta="+str(dp-d0)+", "+str(dp))
  else: pass

print("Final rdm elements: ")
print(r)
print("Final parameters: ")
for i in range(w.shape[0]):
  w[i,:]/=np.sqrt(np.dot(w[i,:],w[i,:]))
print(w)
