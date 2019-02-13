#BUILD IAOS AND COLLECT MOS
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from crystal2pyscf import crystal2pyscf_cell
from pyscf2qwalk import print_qwalk_pbc
from basis import basis, minbasis, basis_order
from pyscf import lo

###########################################################################################
#Build IAO 
mo=None
#for direc in ['../../PBE0/CHK','../../PBE0/COL']:
for direc in ['../../PBE0/FLP']:
  cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=direc+"/GRED.DAT",kred=direc+"/KRED.DAT",totspin=1)
  if(mo is None): mo=mf.mo_coeff[:,:,:,:67]
  else: mo=np.concatenate((mo,mf.mo_coeff[:,:,:,:67]),axis=3)

mf.mo_coeff=np.zeros(mo.shape)
mf.mo_coeff=mo
print(mf.mo_coeff.shape)
print_qwalk_pbc(cell,mf,basename='all_flp')

'''
mo=np.concatenate((mo[0][0],mo[1][0]),axis=1)
s=mf.get_ovlp()[0]
a=lo.iao.iao(cell,mo,minao=minbasis)
a=lo.vec_lowdin(a,s)
print(a.shape)
a.dump('iao.pickle')
'''
'''
mf.mo_coeff=np.zeros((2,1,a.shape[0],a.shape[1]))
mf.mo_coeff[0,0,:,:]=a
print_qwalk_pbc(cell,mf,basename='iao')
'''
