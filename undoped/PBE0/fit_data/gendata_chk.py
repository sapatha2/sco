#Generate excitations, energies and rdms for excitations built on CHK state
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from crystal2pyscf import crystal2pyscf_cell
from pyscf2qwalk import print_qwalk_mol
from methods import calcIAO, rdmIAO, hIAO, genex, data_from_ex, getn, nsum
from basis import basis, minbasis, basis_order

###########################################################################################
#Build IAO 
direc="../CHK"
occ=[i for i in range(72)]
cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=direc+"/GRED.DAT",kred=direc+"/KRED.DAT",totspin=0)
a=calcIAO(cell,mf,minbasis,occ)

#Build excitations
occ=np.arange(24,66)
virt=np.arange(66,72)
ex='singles'
ex_list=genex(mf.mo_occ,[occ,occ],[virt,virt],ex=ex)
e_list,rdm_list=data_from_ex(mf,a,ex_list)
print(ex_list.shape,e_list.shape,rdm_list.shape)

#Analysis
n=getn(rdm_list)
nsum=nsum(n)

for i in range(nsum.shape[0]):
  plt.plot(nsum[i,:],'k.')
plt.show()
