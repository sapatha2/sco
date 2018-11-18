from pyscf import lo
import numpy as np 
from functools import reduce

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

def genex(mo_occ,occ,virt,ex='singles'):
  '''
  generates an excitation list 
  input: 
  mo_occ - molecular orbital occupancy of base state
  occ - a list of occupied orbitals 
  virt - a list of virtual orbitals 
  ex - excitation type
  output: 
  ex_list - list of mo_occ objects of type ex
  '''
  if(ex=='singles'): ex_list=gensingles(mo_occ,occ,virt)
  else: pass
  return ex_list

def gensingles(mo_occ,occ,virt):
  '''
  generates a singles excitation list
  input: 
  mo_occ - molecular orbital occupancy of base state
  occ - a list of occupied orbitals 
  virt - a list of virtual orbitals 
  output: 
  ex_list - list of mo_occ objects for singles excitations
  '''
  ex_list=[]
  ex_list.append(mo_occ)
  print(len(occ),len(virt))
  for j in occ:
    for k in virt:
      tmp=np.array(mo_occ,copy=True)
      assert(tmp[j]==1) #Make sure this is occupied
      assert(tmp[k]==0) #Make sure this is unoccupied
      tmp[j]=0
      tmp[k]=1
      ex_list.append(tmp)
  return np.array(ex_list)

def data_from_ex(mf,ex_list):
  e_list=[np.zeros(len(ex_list[0])),np.zeros(len(ex_list[1]))]
  for s in [0,1]:  
    for i in range(len(ex_list[s])):
      e_list[s][i]=np.sum(mf.mo_energy[s][0][ex_list[s][i].astype(bool)])

  return np.array(e_list),0
