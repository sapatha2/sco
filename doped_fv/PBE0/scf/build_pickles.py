#BUILD IAOS AND COLLECT MOS
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from crystal2pyscf import crystal2pyscf_cell
from pyscf2qwalk import print_qwalk_pbc
from basis import basis, minbasis, basis_order
from pyscf import lo

###########################################################################################
#Build MO pickles
mo=None
active={'FLP':[np.arange(67,73)-1,np.arange(66,73)-1],
        'COL':[np.arange(67,73)-1,np.arange(66,73)-1],
        'COL2':[np.arange(67,73)-1,np.arange(66,73)-1],
        'FM':[np.arange(68,73)-1,np.arange(65,73)-1]}
totspin={'FLP':1,'COL':1,'COL2':1,'FM':3,'UNPOL':1}
#for direc in ['FLP','COL','COL2','FM']:
for direc in ['UNPOL']:
  #d='../'+direc+'_ns'
  d='../'+direc
  cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=d+"/GRED.DAT",kred=d+"/KRED.DAT",totspin=totspin[direc])
  mf.mo_occ[:,0,:].dump('pickles/'+str(direc)+'_mo_occ_g.pickle')
  mf.mo_coeff[:,0,:,:].dump('pickles/'+str(direc)+'_mo_coeff_g.pickle')
  mf.mo_energy[:,0,:].dump('pickles/'+str(direc)+'_mo_energy_g.pickle')
  mf.get_ovlp()[0].dump('pickles/'+str(direc)+'_s_g.pickle')
  
  exit(0)
  new_mo=np.concatenate((mf.mo_coeff[0][0][:,active[direc][0]],mf.mo_coeff[1][0][:,active[direc][1]]),axis=1)
  if(mo is None): mo=new_mo
  else: mo=np.concatenate((mo,new_mo),axis=1)

#Build IAO pickle
print(mo.shape)
s=mf.get_ovlp()[0]
print(s.shape)
a=lo.iao.iao(cell,mo,minao=minbasis)
a=lo.vec_lowdin(a,s)
print(a.shape)
a.dump('pickles/iao_g.pickle')

#Plot IAOs
a=np.load('pickles/iao_g.pickle')
mf.mo_coeff=np.zeros((2,1,a.shape[0],a.shape[1]))
mf.mo_coeff[0,0,:,:]=a
print_qwalk_pbc(cell,mf,basename='orbs/iao_g')
