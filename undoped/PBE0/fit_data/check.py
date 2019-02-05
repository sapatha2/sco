#Check for active space, IAO basis, etc
import numpy as np 
import matplotlib.pyplot as plt
from crystal2pyscf import crystal2pyscf_cell
from methods import calcIAO, rdmIAO, hIAO
from basis import basis, minbasis, basis_order

###########################################################################################
#Build IAO 
direc="../CHK"
occ=[i for i in range(72)]
cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=direc+"/GRED.DAT",kred=direc+"/KRED.DAT",totspin=0)
a=calcIAO(cell,mf,minbasis,occ)

#Import states
direc="../FM"
cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=direc+"/GRED.DAT",kred=direc+"/KRED.DAT",totspin=4)

#Build matrices
h=hIAO(mf,a)
rdm=rdmIAO(mf,a)

#Check matrices
print(np.trace(rdm[0]),np.trace(rdm[1]),132-np.trace(rdm[0])-np.trace(rdm[1]))
w,vr=np.linalg.eigh(h[0])
plt.plot(np.arange(1,len(w)+1),mf.mo_energy[0][0][:len(w)],'go')
plt.plot(np.arange(1,len(w)+1),w,'bo')
w,vr=np.linalg.eigh(h[1])
plt.plot(np.arange(1,len(w)+1),mf.mo_energy[1][0][:len(w)],'r*')
plt.plot(np.arange(1,len(w)+1),w,'k*')
plt.show()
