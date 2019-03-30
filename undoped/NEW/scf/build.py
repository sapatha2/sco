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
active={'CHK':[67,67],'COL':[67,67],'FLP':[67,66],'FM':[68,65]} #Active plus first excitation
totspin={'CHK':0,'COL':0,'FLP':2,'FM':4}
for direc in ['CHK']:#,'COL','FLP','FM']:
  d='../../PBE0/'+direc
  cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=d+"/GRED.DAT",kred=d+"/KRED.DAT",totspin=totspin[direc])
  new_mo=np.concatenate((mf.mo_coeff[0][0][:,:active[direc][0]],mf.mo_coeff[1][0][:,:active[direc][1]]),axis=1)
  if(mo is None): mo=new_mo
  else: mo=np.concatenate((mo,new_mo),axis=1)
  #if(mo is None): mo=mf.mo_coeff[:,:,:,:69]
  #else: mo=np.concatenate((mo,mf.mo_coeff[:,:,:,:69]),axis=3)

#mf.mo_coeff=np.zeros(mo.shape)
#mf.mo_coeff=mo
#print(mf.mo_coeff.shape)
#print_qwalk_pbc(cell,mf,basename='all')

#mo=np.concatenate((mo[0][0],mo[1][0]),axis=1)
print(mo.shape)
s=mf.get_ovlp()[0]
a=lo.iao.iao(cell,mo,minao=minbasis)
a=lo.vec_lowdin(a,s)
print(a.shape)
a.dump('iao.pickle')

#a=np.load('iao.pickle')
#mf.mo_coeff=np.zeros((2,1,a.shape[0],a.shape[1]))
#mf.mo_coeff[0,0,:,:]=a
#print_qwalk_pbc(cell,mf,basename='orbs/iao')
