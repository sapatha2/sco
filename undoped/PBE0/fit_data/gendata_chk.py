#Generate excitations, energies and rdms for excitations built on CHK state
import numpy as np 
import matplotlib.pyplot as plt
from crystal2pyscf import crystal2pyscf_cell
from methods import calcIAO, rdmIAO, hIAO, genex, data_from_ex
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
ex_list=[genex(mf.mo_occ[0][0],occ,virt,ex='singles'),
         genex(mf.mo_occ[1][0],occ,virt,ex='singles')]
e_list,rdm_list=data_from_ex(mf,ex_list)

#Check
plt.hist((e_list[0]-e_list[0][0])*27.2114,bins=20)
plt.show()
#rdm=rdmIAO(mf,a)
