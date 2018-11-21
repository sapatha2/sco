from pyscf import lo
import numpy as np 
from functools import reduce

#BASE
def calcIAO(cell,mf,basis,occ):
  ''' 
  input: 
  cell and scf (PBC SCF) objects from pyscf and basis
  to calculate IAOs on
  occupied MOs which IAOs should span
  output:
  Calculates 1RDM on orthog atomic orbitals, orthogonalized
  using Lowdin S^1/2
  Returns coefficient matrix for IAOs 
  '''
  s=mf.get_ovlp()[0]

  mo_occ = mf.mo_coeff[0][0][:,occ]
  mo_occ2 = mf.mo_coeff[1][0][:,occ]
  mo_occ=np.concatenate((mo_occ,mo_occ2),axis=1)
  a = lo.iao.iao(cell, mo_occ, minao=basis)
  a = lo.vec_lowdin(a, s)
 
  return a

def printIAO(mf,a):
  for i in range(a.shape[1]):
    mf.mo_coeff[0][0][:,i]=a[:,i]
  print_qwalk_mol(cell,mf,basename="./iao/qwalk_iao")

#SINGLE STATE CALCULATION
def rdmIAO(mf,a):
  ''' 
  input:
  mf object for calculation
  output:
  1rdm for spin up and down 
  '''
  s=mf.get_ovlp()[0]
  mo_occ = mf.mo_coeff[0][0][:,mf.mo_occ[0][0]>0]
  mo_occ = reduce(np.dot, (a.T, s, mo_occ))
  dm_u = np.dot(mo_occ, mo_occ.T)
  mo_occ = mf.mo_coeff[1][0][:,mf.mo_occ[1][0]>0]
  mo_occ = reduce(np.dot, (a.T, s, mo_occ))
  dm_d = np.dot(mo_occ, mo_occ.T)
  return np.array([dm_u,dm_d])

def hIAO(mf,a):
  '''
  input: 
  mf object for calculation
  IAO vector a
  output: 
  eigenvalue matrix in IAO basis 
  '''
  s=mf.get_ovlp()[0]
  H1=np.diag(mf.mo_energy[0][0])
  e1u=reduce(np.dot,(a.T,s,mf.mo_coeff[0][0],H1,mf.mo_coeff[0][0].T,s.T,a))
  e1u=(e1u+e1u.T)/2
  H1=np.diag(mf.mo_energy[1][0])
  e1d=reduce(np.dot,(a.T,s,mf.mo_coeff[1][0],H1,mf.mo_coeff[1][0].T,s.T,a))
  e1d=(e1d+e1d.T)/2
  return np.array([e1u,e1d])

#EXCITATION CALCULATIONS
def genex(mo_occ,occ,virt,ex='singles'):
  '''
  generates an excitation list 
  input: 
  mo_occ - molecular orbital occupancy of base state
  occ - a list of occupied orbitals in up and down channels (occ[0],occ[1])
  virt - a list of virtual orbitals in up and down channels (occ[0],occ[1])
  ex - excitation type
  output: 
  ex_list - list of mo_occ objects of type ex
  '''
  print("Generating excitations") 
  if(ex=='singles'): ex_list=gensingles(mo_occ,occ,virt)
  else: pass
  return ex_list

def gensingles(mo_occ,occ,virt):
  '''
  generates a singles excitation list
  input: 
  mo_occ - molecular orbital occupancy of base state
  occ - a list of occupied orbitals in up and down channels (occ[0],occ[1])
  virt - a list of virtual orbitals in up and down channels (occ[0],occ[1])
  output: 
  ex_list - list of mo_occ objects for singles excitations
  '''
  print("Singles excitations")
  ex_list=[]
  ex_list.append(mo_occ)
  for s in [0,1]:
    for j in occ[s]:
      for k in virt[s]:
        tmp=np.array(mo_occ,copy=True)
        assert(tmp[s][0][j]==1)
        assert(tmp[s][0][k]==0)
        tmp[s][0][j]=0
        tmp[s][0][k]=1
        ex_list.append(tmp)
  return np.array(ex_list)

def data_from_ex(mf,a,ex_list):
  '''
  generates energies and 1rdms from excitation list provided
  input: 
  mf - scf object of base state
  a - basis to calculate 1RDM on 
  ex_list - excitation list of states to calculate for
  output:
  e - list of energies for excitations 
  dm - list of 1rdms (spin separated) for excitations
  '''
  
  print("Generating energies")
  el=np.einsum('ijk,jk->i',ex_list[:,:,0,:],mf.mo_energy[:,0,:])

  print("Generating 1rdms")
  s=mf.get_ovlp()[0]
  M=reduce(np.dot,(a.T, s, mf.mo_coeff[0][0])) 
  R=np.einsum('ik,jk->ijk',ex_list[:,0,0,:],M)
  dm_u=np.einsum('ijl,ikl->ijk',R,R)
  M=reduce(np.dot,(a.T, s, mf.mo_coeff[1][0])) 
  R=np.einsum('ik,jk->ijk',ex_list[:,1,0,:],M)
  dm_d=np.einsum('ijl,ikl->ijk',R,R)
  dm=np.einsum('ijkl->jikl',np.array([dm_u,dm_d]))

  return el-el[0],dm

#OCCUPATIONS
def getn(dm_list):
  '''
  returns diagonals of dm_list
  input: 
  dm_list - list of 1rdms
  output:
  n - list of diagonals of 1rdms
  '''
  return np.einsum('ijkk->ijk',dm_list)

def nsum(n):
  '''
  takes output from getn, groups by symmetry
  input: 
  n - output from get n 
  output: 
  nsum - returns sum of n based on symmetry
  '''
  sr5s=  np.arange(4)
  o2s=   np.arange(44,76,4)
  o2pz=  np.arange(47,76,4)
  o2psg= np.array([45,49,53,57,62,66,70,74]) 
  o2ppi= np.array([46,50,54,58,61,65,69,73])
  cu4s1= np.array([5,35])  #Occupation of spin parallel to local moment
  cu4s2= np.array([15,25]) #Occupation of spin antiparallel to local moment 

  nsum_u=np.zeros((n.shape[0],17))
  nsum_d=np.zeros((n.shape[0],17))
  T=np.zeros((17,n.shape[2]))
  T[0,:][sr5s]=1./4
  T[1,:][o2s]=1./8
  T[2,:][o2psg]=1./8
  T[3,:][o2ppi]=1./8
  T[4,:][o2pz]=1./8
  R=np.array(T,copy=True)
  
  T[5,:][cu4s1]=1./4
  T[11,:][cu4s2]=1./4
  for i in range(6,11):
    T[i,:][cu4s1+i-2]=1./4
    T[i+6,:][cu4s2+i-2]=1./4
  nsum_u=np.einsum('ik,lk->il',n[:,0,:],T)

  R[5,:][cu4s2]=1./4
  R[11,:][cu4s1]=1./4
  for i in range(6,11):
    R[i,:][cu4s2+i-2]=1./4
    R[i+6,:][cu4s1+i-2]=1./4
  nsum_d=np.einsum('ik,lk->il',n[:,1,:],R)

  return nsum_u+nsum_d

def Usum(n):
  Usig=n[:,0,:]*n[:,1,:]
  sr5s=  np.arange(4)
  o2s=   np.arange(44,76,4)
  o2pz=  np.arange(47,76,4)
  o2psg= np.array([45,49,53,57,62,66,70,74]) 
  o2ppi= np.array([46,50,54,58,61,65,69,73])
  cu4s= np.array([5,15,25,35])  #Occupation of spin parallel to local moment
  
  T=np.zeros((11,n.shape[2]))
  T[0,:][sr5s]=1.
  T[1,:][o2s]=1.
  T[2,:][o2psg]=1.
  T[3,:][o2ppi]=1.
  T[4,:][o2pz]=1.
  T[5,:][cu4s]=1.
  for i in range(6,11):
    T[i,:][cu4s+i-2]=1.
  ret=np.einsum('ik,jk->ij',Usig,T)
  return ret

#HOPPINGS
def ts(dm_list,orbital,n):
  olist=[[44,60,52,64],
         [48,64,56,60],
         [52,72,44,68],
         [56,68,48,72]]
  if(orbital=="4s"): 
    culist=[5,15,25,35]
    sign=[1,1,1,1]
  elif(orbital=="3dz2"): 
    culist=[11,21,31,41]
    sign=[1,1,1,1]
  elif(orbital=="3d"): 
    culist=[13,23,33,43]
    sign=[1,-1,1,-1]
  else: 
    print("Hopping for orbital "+orbital+" not implemented")
  
  if(n==1): sw=[0,3,1,2]
  else: sw=[1,2,0,3]
  
  T=np.zeros(dm_list.shape[2:])
  T[culist[sw[0]],olist[sw[0]]]=sign
  T[culist[sw[1]],olist[sw[1]]]=sign
  T+=T.T
  t_u=np.einsum('ikl,kl->i',dm_list[:,0,:,:],T)
  
  T=np.zeros(dm_list.shape[2:])
  T[culist[sw[2]],olist[sw[2]]]=sign
  T[culist[sw[3]],olist[sw[3]]]=sign
  T+=T.T
  t_d=np.einsum('ikl,kl->i',dm_list[:,1,:,:],T)
  
  return t_u + t_d
 
def tsig(dm_list,orbital,n):
  olist=[[45,62,53,66],
         [49,66,57,62],
         [53,74,45,70],
         [57,70,49,74]]
  if(orbital=="4s"): 
    culist=[5,15,25,35]
    sign=[-1,-1,1,1]
  elif(orbital=="3dz2"): 
    culist=[11,21,31,41]
    sign=[-1,-1,1,1]
  elif(orbital=="3d"): 
    culist=[13,23,33,43]
    sign=[-1,1,1,-1]
  else: 
    print("Hopping for orbital "+orbital+" not implemented")
    return -1

  if(n==1): sw=[0,3,1,2]
  else: sw=[1,2,0,3]

  T=np.zeros(dm_list.shape[2:])
  T[culist[sw[0]],olist[sw[0]]]=sign
  T[culist[sw[1]],olist[sw[1]]]=sign
  T+=T.T
  t_u=np.einsum('ikl,kl->i',dm_list[:,0,:,:],T)

  T=np.zeros(dm_list.shape[2:])
  T[culist[sw[2]],olist[sw[2]]]=sign
  T[culist[sw[3]],olist[sw[3]]]=sign
  T+=T.T
  t_d=np.einsum('ikl,kl->i',dm_list[:,1,:,:],T)

  return t_u + t_d

def tpi(dm_list,orbital,n):
  olist=[[46,61,54,65],
         [50,65,58,61],
         [54,73,46,69],
         [58,69,50,73]]
  if(orbital=="3dxy"): 
    culist=[9,19,29,39]
    sign=[1,1,-1,-1]
  else: 
    print("Hopping for orbital "+orbital+" not implemented")
    return -1

  if(n==1): sw=[0,3,1,2]
  else: sw=[1,2,0,3]

  T=np.zeros(dm_list.shape[2:])
  T[culist[sw[0]],olist[sw[0]]]=sign
  T[culist[sw[1]],olist[sw[1]]]=sign
  T+=T.T
  t_u=np.einsum('ikl,kl->i',dm_list[:,0,:,:],T)

  T=np.zeros(dm_list.shape[2:])
  T[culist[sw[2]],olist[sw[2]]]=sign
  T[culist[sw[3]],olist[sw[3]]]=sign
  T+=T.T
  t_d=np.einsum('ikl,kl->i',dm_list[:,1,:,:],T)

  return t_u + t_d

def tz(dm_list,orbital,n):
  olist=[[47,55],
         [51,59],
         [55,47],
         [59,51]]
  if(orbital=="3dxz"): 
    culist=[12,22,32,42]
    sign=[1,-1]
  else: 
    print("Hopping for orbital "+orbital+" not implemented")
    return -1

  if(n==1): sw=[0,3,1,2]
  else: sw=[1,2,0,3]

  T=np.zeros(dm_list.shape[2:])
  T[culist[sw[0]],olist[sw[0]]]=sign
  T[culist[sw[1]],olist[sw[1]]]=sign
  T+=T.T
  t_u=np.einsum('ikl,kl->i',dm_list[:,0,:,:],T)

  T=np.zeros(dm_list.shape[2:])
  T[culist[sw[2]],olist[sw[2]]]=sign
  T[culist[sw[3]],olist[sw[3]]]=sign
  T+=T.T
  t_d=np.einsum('ikl,kl->i',dm_list[:,1,:,:],T)

  return t_u + t_d

def too(dm_list,a,b):
  onn=np.array([[7,8,5,6],
       [8,7,6,5],
       [6,5,8,7],
       [5,6,7,8],
       [1,2,3,4],
       [2,1,3,4],
       [4,3,1,2],
       [3,4,2,1]])-1

  oos=np.array([[44,45,46,47],
       [48,49,50,51],
       [52,53,54,55],
       [56,57,58,59],
       [60,62,61,63],
       [64,66,65,67],
       [68,70,69,71],
       [72,74,73,75]])
  
  T=np.zeros(dm_list.shape[2:])
  if(a=="sig" and b=="sig"):
    sign=[1,-1,1,-1]
    for i in range(8):
      T[oos[i][1],oos[onn[i],1]]=sign
    T+=T.T
  elif(a=="pi" and b=="pi"):
    sign=[1,-1,1,-1]
    for i in range(8):
      T[oos[i][2],oos[onn[i],2]]=sign
    T+=T.T
  elif(a=="z" and b=="z"):
    sign=1
    for i in range(8):
      T[oos[i][3],oos[onn[i],3]]=sign
    T+=T.T
  elif(a=="s" and b=="s"):
    sign=1
    for i in range(8):
      T[oos[i][0],oos[onn[i],0]]=sign
    T+=T.T
  elif(a=="sig" and b=="pi"):
    sign=-1
    for i in range(8):
      T[oos[i][1],oos[onn[i],2]]=sign
    T+=T.T

  t_u=np.einsum('ikl,kl->i',dm_list[:,0,:,:],T)
  t_d=np.einsum('ikl,kl->i',dm_list[:,1,:,:],T)

  return t_u + t_d
