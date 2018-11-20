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
  e=np.einsum('ijk,jk->i',ex_list[:,:,0,:],mf.mo_energy[:,0,:])
  
  print("Generating 1rdms")
  s=mf.get_ovlp()[0]
  M=reduce(np.dot,(a.T, s, mf.mo_coeff[0][0])) 
  R=np.einsum('ik,jk->ijk',ex_list[:,0,0,:],M)
  dm_u=np.einsum('ijl,ikl->ijk',R,R)
  M=reduce(np.dot,(a.T, s, mf.mo_coeff[1][0])) 
  R=np.einsum('ik,jk->ijk',ex_list[:,1,0,:],M)
  dm_d=np.einsum('ijl,ikl->ijk',R,R)
  dm=np.einsum('ijkl->jikl',np.array([dm_u,dm_d]))

  return e,dm

#OCCUPATIONS
def getn(dm_list):
  '''
  returns diagonals of dm_list
  intput: 
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

#HOPPINGS
def tsig3d2(dm_list):
  '''
  calculates hopping between 2psig and 3dx2y2 2
  '''
  #Up channel
  T=np.zeros(dm_list.shape[2:])
  T[23,[49,57,62,66]]=[-1,1,-1,1]
  T[33,[45,53,70,74]]=[1,-1,1,-1]
  tsig_u=np.einsum('ikl,kl->i',dm_list[:,0,:,:],T)*2

  #Down channel
  T=np.zeros(dm_list.shape[2:])
  T[13,[45,53,62,66]]=[-1,1,1,-1]
  T[43,[49,57,70,74]]=[1,-1,-1,1]
  tsig_d=np.einsum('ikl,kl->i',dm_list[:,1,:,:],T)*2

  return tsig_u+tsig_d

def tsig3d1(dm_list):
  '''
  calculates hopping between 2psig and 3dx2y2 2
  '''
  #Up channel
  T=np.zeros(dm_list.shape[2:])
  T[13,[45,53,62,66]]=[-1,1,1,-1]
  T[43,[49,57,70,74]]=[1,-1,-1,1]
  tsig_u=np.einsum('ikl,kl->i',dm_list[:,0,:,:],T)*2

  #Down channel
  T=np.zeros(dm_list.shape[2:])
  T[23,[49,57,62,66]]=[-1,1,-1,1]
  T[33,[45,53,70,74]]=[1,-1,1,-1]
  tsig_d=np.einsum('ikl,kl->i',dm_list[:,1,:,:],T)*2

  return tsig_u+tsig_d

def tsig4s2(dm_list):
  '''
  calculates hopping between 2psig and 3dx2y2 2
  '''
  #Up channel
  T=np.zeros(dm_list.shape[2:])
  T[15,[49,57,62,66]]=[-1,1,1,-1]
  T[25,[45,53,70,74]]=[1,-1,-1,1]
  tsig_u=np.einsum('ikl,kl->i',dm_list[:,0,:,:],T)*2

  #Down channel
  T=np.zeros(dm_list.shape[2:])
  T[5,[45,53,62,66]]=[-1,1,-1,1]
  T[35,[49,57,70,74]]=[1,-1,1,-1]
  tsig_d=np.einsum('ikl,kl->i',dm_list[:,1,:,:],T)*2

  return tsig_u+tsig_d

def tsig4s1(dm_list):
  '''
  calculates hopping between 2psig and 3dx2y2 2
  '''
  #Up channel
  T=np.zeros(dm_list.shape[2:])
  T[5,[45,53,62,66]]=[-1,1,-1,1]
  T[35,[49,57,70,74]]=[1,-1,1,-1]
  tsig_u=np.einsum('ikl,kl->i',dm_list[:,0,:,:],T)*2

  #Down channel
  T=np.zeros(dm_list.shape[2:])
  T[15,[49,57,62,66]]=[-1,1,1,-1]
  T[25,[45,53,70,74]]=[1,-1,-1,1]
  tsig_d=np.einsum('ikl,kl->i',dm_list[:,1,:,:],T)*2

  return tsig_u+tsig_d

def tsig3dz2(dm_list):
  '''
  calculates hopping between 2psig and 3dx2y2 2
  '''
  #Up channel
  T=np.zeros(dm_list.shape[2:])
  T[21,[49,57,62,66]]=[-1,1,1,-1]
  T[31,[45,53,70,74]]=[1,-1,-1,1]
  tsig_u=np.einsum('ikl,kl->i',dm_list[:,0,:,:],T)*2

  #Down channel
  T=np.zeros(dm_list.shape[2:])
  T[11,[45,53,62,66]]=[-1,1,-1,1]
  T[41,[49,57,70,74]]=[1,-1,1,-1]
  tsig_d=np.einsum('ikl,kl->i',dm_list[:,1,:,:],T)*2

  return tsig_u+tsig_d

def tsig3dz1(dm_list):
  '''
  calculates hopping between 2psig and 3dx2y2 2
  '''
  #Up channel
  T=np.zeros(dm_list.shape[2:])
  T[11,[45,53,62,66]]=[-1,1,-1,1]
  T[41,[49,57,70,74]]=[1,-1,1,-1]
  tsig_u=np.einsum('ikl,kl->i',dm_list[:,0,:,:],T)*2

  #Down channel
  T=np.zeros(dm_list.shape[2:])
  T[21,[49,57,62,66]]=[-1,1,1,-1]
  T[31,[45,53,70,74]]=[1,-1,-1,1]
  tsig_d=np.einsum('ikl,kl->i',dm_list[:,1,:,:],T)*2

  return tsig_u+tsig_d

def ts3d2(dm_list):
  '''
  calculates hopping between 2psig and 3dx2y2 2
  '''
  #Up channel
  T=np.zeros(dm_list.shape[2:])
  T[23,[48,56,60,64]]=[1,1,-1,-1]
  T[33,[44,52,68,72]]=[1,1,-1,-1]
  tsig_u=np.einsum('ikl,kl->i',dm_list[:,0,:,:],T)*2

  #Down channel
  T=np.zeros(dm_list.shape[2:])
  T[13,[44,52,60,64]]=[1,1,-1,-1]
  T[43,[48,56,68,72]]=[1,1,-1,-1]
  tsig_d=np.einsum('ikl,kl->i',dm_list[:,1,:,:],T)*2

  return tsig_u+tsig_d

def ts3d1(dm_list):
  '''
  calculates hopping between 2psig and 3dx2y2 2
  '''
  #Up channel
  T=np.zeros(dm_list.shape[2:])
  T[13,[44,52,60,64]]=[1,1,-1,-1]
  T[43,[48,56,68,72]]=[1,1,-1,-1]
  tsig_u=np.einsum('ikl,kl->i',dm_list[:,0,:,:],T)*2

  #Down channel
  T=np.zeros(dm_list.shape[2:])
  T[23,[48,56,60,64]]=[1,1,-1,-1]
  T[33,[44,52,68,72]]=[1,1,-1,-1]
  tsig_d=np.einsum('ikl,kl->i',dm_list[:,1,:,:],T)*2

  return tsig_u+tsig_d

def ts4s2(dm_list):
  '''
  calculates hopping between 2psig and 3dx2y2 2
  '''
  #Up channel
  T=np.zeros(dm_list.shape[2:])
  T[15,[48,56,60,64]]=1
  T[25,[44,52,68,72]]=1
  tsig_u=np.einsum('ikl,kl->i',dm_list[:,0,:,:],T)*2

  #Down channel
  T=np.zeros(dm_list.shape[2:])
  T[5,[44,52,60,64]]=1
  T[35,[48,56,68,72]]=1
  tsig_d=np.einsum('ikl,kl->i',dm_list[:,1,:,:],T)*2

  return tsig_u+tsig_d

def ts4s1(dm_list):
  '''
  calculates hopping between 2psig and 3dx2y2 2
  '''
  #Up channel
  T=np.zeros(dm_list.shape[2:])
  T[5,[44,52,60,64]]=1
  T[35,[48,56,68,72]]=1
  tsig_u=np.einsum('ikl,kl->i',dm_list[:,0,:,:],T)*2

  #Down channel
  T=np.zeros(dm_list.shape[2:])
  T[15,[48,56,60,64]]=1
  T[25,[44,52,68,72]]=1
  tsig_d=np.einsum('ikl,kl->i',dm_list[:,1,:,:],T)*2

  return tsig_u+tsig_d

def ts3dz2(dm_list):
  '''
  calculates hopping between 2psig and 3dx2y2 2
  '''
  #Up channel
  T=np.zeros(dm_list.shape[2:])
  T[21,[48,56,60,64]]=1
  T[31,[44,52,68,72]]=1
  tsig_u=np.einsum('ikl,kl->i',dm_list[:,0,:,:],T)*2

  #Down channel
  T=np.zeros(dm_list.shape[2:])
  T[11,[44,52,60,64]]=1
  T[41,[48,56,68,72]]=1
  tsig_d=np.einsum('ikl,kl->i',dm_list[:,1,:,:],T)*2

  return tsig_u+tsig_d

def ts3dz1(dm_list):
  '''
  calculates hopping between 2psig and 3dx2y2 2
  '''
  #Up channel
  T=np.zeros(dm_list.shape[2:])
  T[11,[44,52,60,64]]=1
  T[41,[48,56,68,72]]=1
  tsig_u=np.einsum('ikl,kl->i',dm_list[:,0,:,:],T)*2

  #Down channel
  T=np.zeros(dm_list.shape[2:])
  T[21,[48,56,60,64]]=1
  T[31,[44,52,68,72]]=1
  tsig_d=np.einsum('ikl,kl->i',dm_list[:,1,:,:],T)*2

  return tsig_u+tsig_d

def tp3dxy2(dm_list):
  '''
  calculates hopping between 2psig and 3dx2y2 2
  '''
  #Up channel
  T=np.zeros(dm_list.shape[2:])
  T[19,[50,58,61,65]]=[1,-1,-1,1]
  T[29,[46,54,69,73]]=[-1,1,1,-1]
  tsig_u=np.einsum('ikl,kl->i',dm_list[:,0,:,:],T)*2

  #Down channel
  T=np.zeros(dm_list.shape[2:])
  T[9,[46,54,61,65]]=[1,-1,1,-1]
  T[39,[48,58,69,73]]=[-1,1,-1,1]
  tsig_d=np.einsum('ikl,kl->i',dm_list[:,1,:,:],T)*2

  return tsig_u+tsig_d

def tp3dxy1(dm_list):
  '''
  calculates hopping between 2psig and 3dx2y2 2
  '''
  #Up channel
  T=np.zeros(dm_list.shape[2:])
  T[9,[46,54,61,65]]=[1,-1,1,-1]
  T[39,[48,58,69,73]]=[-1,1,-1,1]
  tsig_u=np.einsum('ikl,kl->i',dm_list[:,0,:,:],T)*2

  #Down channel
  T=np.zeros(dm_list.shape[2:])
  T[19,[50,58,61,65]]=[1,-1,-1,1]
  T[29,[46,54,69,73]]=[-1,1,1,-1]
  tsig_d=np.einsum('ikl,kl->i',dm_list[:,1,:,:],T)*2

  return tsig_u+tsig_d

def tz3dyz2(dm_list):
  '''
  calculates hopping between 2psig and 3dx2y2 2
  '''
  #Up channel
  T=np.zeros(dm_list.shape[2:])
  T[20,[63,67]]=[1,-1]
  T[30,[71,75]]=[-1,1]
  tsig_u=np.einsum('ikl,kl->i',dm_list[:,0,:,:],T)*2

  #Down channel
  T=np.zeros(dm_list.shape[2:])
  T[10,[63,67]]=[-1,1]
  T[40,[71,75]]=[1,-1]
  tsig_d=np.einsum('ikl,kl->i',dm_list[:,1,:,:],T)*2

  return tsig_u+tsig_d

def tz3dyz1(dm_list):
  '''
  calculates hopping between 2psig and 3dx2y2 2
  '''
  #Up channel
  T=np.zeros(dm_list.shape[2:])
  T[10,[63,67]]=[-1,1]
  T[40,[71,75]]=[1,-1]
  tsig_u=np.einsum('ikl,kl->i',dm_list[:,0,:,:],T)*2

  #Down channel
  T=np.zeros(dm_list.shape[2:])
  T[20,[63,67]]=[1,-1]
  T[30,[71,75]]=[-1,1]
  tsig_d=np.einsum('ikl,kl->i',dm_list[:,1,:,:],T)*2

  return tsig_u+tsig_d
