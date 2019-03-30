#COLLECT T, U, J FOR MY 4 RELEVANT STATES
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from crystal2pyscf import crystal2pyscf_cell
from pyscf2qwalk import print_qwalk_pbc
from basis import basis, minbasis, basis_order
from pyscf import lo
from functools import reduce 
from downfold_tools import gen_slater_tbdm,sum_onebody,sum_J,sum_U
import seaborn as sns 
from copy import deepcopy

df=None
direcs=['../../PBE0/CHK']*2+['../../PBE0/COL']*1+['../../PBE0/FLP']*1+['../../PBE0/FM']*1
totspin=[0]*2+[1]*1+[2]*1+[4]*1
base_energy=[-9.2123749620223E+02]*2+[-9.2122638412999E+02]*1+[-9.2122636552756E+02]*1+[-9.2121137910381E+02]*1
rem=[[[],[]],
     [[],[65]],
     [[],[]],
     [[],[]],
     [[],[]]]
add=[[[],[]], 
     [[],[66]],
     [[],[]],
     [[],[]],
     [[],[]]]
for run in range(len(totspin)):
  direc=direcs[run]
  cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=direc+"/GRED.DAT",kred=direc+"/KRED.DAT",totspin=totspin[run])
  s=mf.get_ovlp()
  
  if(run==0): a=deepcopy(mf.mo_coeff[1][0][:,[65,66]])
  print(a.shape)

  obdm=np.zeros((2,a.shape[1],a.shape[1]))
  e=base_energy[run]
  for spin in [0,1]:
    mo_rdm=mf.mo_occ[spin][0]
    mo_to_iao=reduce(np.dot,(a.T, s, mf.mo_coeff[spin][0]))
    if(len(rem[run][spin])>0):
      mo_rdm[rem[run][spin]]=0
      mo_rdm[add[run][spin]]=1
      e+=(mf.mo_energy[spin][0][add[run][spin]]-mf.mo_energy[spin][0][rem[run][spin]])
    obdm[spin]=reduce(np.dot,(mo_to_iao,np.diag(mo_rdm),mo_to_iao.T))
  tbdm,__=gen_slater_tbdm(obdm)
  e*=27.2114
  print(np.trace(obdm[0]),np.trace(obdm[1]),np.trace(obdm[0]+obdm[1]))

  #Hopping
  orb1=[0,1]
  n=sum_onebody(obdm,orb1,orb1)

  d=pd.DataFrame({'energy':[e],'n_0':[n[0]],'n_1':[n[1]]})
  if(df is None): df=d
  else: df=pd.concat((df,d),axis=0)

print(df)
df.to_pickle('scf_mo_gosling.pickle')
