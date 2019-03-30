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

a=np.load('iao.pickle')

df=None
direcs=['../../PBE0/CHK']*73+['../../PBE0/COL']*1+['../../PBE0/FLP']*1+['../../PBE0/FM']*1
totspin=[0]*73+[1]*1+[2]*1+[4]*1
base_energy=[-9.2123749620223E+02]*73+[-9.2122638412999E+02]*1+[-9.2122636552756E+02]*1+[-9.2121137910381E+02]*1
rem=[[[],[]],
     [[24],[]],
     [[25],[]],
     [[26],[]],
     [[28],[]],
     [[36],[]],
     [[37],[]],
     [[44],[]],
     [[45],[]],
     [[46],[]],
     [[47],[]],
     [[51],[]],
     [[55],[]],
     [[24],[]],
     [[25],[]],
     [[26],[]],
     [[28],[]],
     [[36],[]],
     [[37],[]],
     [[44],[]],
     [[45],[]],
     [[46],[]],
     [[47],[]],
     [[51],[]],
     [[55],[]],
     [[24],[]],
     [[25],[]],
     [[26],[]],
     [[28],[]],
     [[36],[]],
     [[37],[]],
     [[44],[]],
     [[45],[]],
     [[46],[]],
     [[47],[]],
     [[51],[]],
     [[55],[]],
     [[24],[]],
     [[25],[]],
     [[26],[]],
     [[28],[]],
     [[36],[]],
     [[37],[]],
     [[44],[]],
     [[45],[]],
     [[46],[]],
     [[47],[]],
     [[51],[]],
     [[55],[]],
     [[24],[]],
     [[25],[]],
     [[26],[]],
     [[28],[]],
     [[36],[]],
     [[37],[]],
     [[44],[]],
     [[45],[]],
     [[46],[]],
     [[47],[]],
     [[51],[]],
     [[55],[]],
     [[24],[]],
     [[25],[]],
     [[26],[]],
     [[28],[]],
     [[36],[]],
     [[37],[]],
     [[44],[]],
     [[45],[]],
     [[46],[]],
     [[47],[]],
     [[51],[]],
     [[55],[]],
     [[],[]],
     [[],[]],
     [[],[]]]
add=[[[],[]],
     [[66],[]],
     [[66],[]],
     [[66],[]],
     [[66],[]],
     [[66],[]],
     [[66],[]],
     [[66],[]],
     [[66],[]],
     [[66],[]],
     [[66],[]],
     [[66],[]],
     [[66],[]],
     [[67],[]],
     [[67],[]],
     [[67],[]],
     [[67],[]],
     [[67],[]],
     [[67],[]],
     [[67],[]],
     [[67],[]],
     [[67],[]],
     [[67],[]],
     [[67],[]],
     [[67],[]],
     [[68],[]],
     [[68],[]],
     [[68],[]],
     [[68],[]],
     [[68],[]],
     [[68],[]],
     [[68],[]],
     [[68],[]],
     [[68],[]],
     [[68],[]],
     [[68],[]],
     [[68],[]],
     [[69],[]],
     [[69],[]],
     [[69],[]],
     [[69],[]],
     [[69],[]],
     [[69],[]],
     [[69],[]],
     [[69],[]],
     [[69],[]],
     [[69],[]],
     [[69],[]],
     [[69],[]],
     [[70],[]],
     [[70],[]],
     [[70],[]],
     [[70],[]],
     [[70],[]],
     [[70],[]],
     [[70],[]],
     [[70],[]],
     [[70],[]],
     [[70],[]],
     [[70],[]],
     [[70],[]],
     [[71],[]],
     [[71],[]],
     [[71],[]],
     [[71],[]],
     [[71],[]],
     [[71],[]],
     [[71],[]],
     [[71],[]],
     [[71],[]],
     [[71],[]],
     [[71],[]],
     [[71],[]],
     [[],[]],
     [[],[]],
     [[],[]]]
for run in range(len(totspin)):
  direc=direcs[run]
  cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=direc+"/GRED.DAT",kred=direc+"/KRED.DAT",totspin=totspin[run])
  s=mf.get_ovlp()
  obdm =np.zeros((2,a.shape[1],a.shape[1]))
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

  #Hopping
  orb1=np.array([14,14,14,14,24,24,24,24,34,34,34,34,44,44,44,44])-1
  orb2=np.array([46,63,54,67,50,67,58,63,54,71,46,75,58,75,50,71])-1
  sign=np.array([-1,1,1,-1]*4)
  sigT=sum_onebody(obdm,orb1,orb2)
  sigT=2*np.dot(sign,sigT)

  orb1=np.array([14,14,24,24,34,34,44,44])-1
  orb2=np.array([24,34,14,44,14,44,24,34])-1
  sigTd=2*np.sum(sum_onebody(obdm,orb1,orb2))

  #Number occupations
  orb1=np.array([6,16,26,36])-1
  sigN4s=np.sum(sum_onebody(obdm,orb1,orb1))

  orb1=np.array([45,49,53,57,61,65,69,73])-1
  sigN2s=np.sum(sum_onebody(obdm,orb1,orb1))

  orb1=np.array([46,50,54,58,63,67,71,75])-1
  sigNps=np.sum(sum_onebody(obdm,orb1,orb1))

  orb1=np.array([47,51,55,59,62,66,70,74])-1
  sigNpp=np.sum(sum_onebody(obdm,orb1,orb1))
  
  orb1=np.array([48,52,56,60,64,68,72,76])-1
  sigNpz=np.sum(sum_onebody(obdm,orb1,orb1))

  orb1=np.array([11,13,21,23,31,33,41,43])-1
  sigNdz=np.sum(sum_onebody(obdm,orb1,orb1))

  orb1=np.array([14,24,34,44])-1
  sigNd=np.sum(sum_onebody(obdm,orb1,orb1))

  orb1=np.array([12,22,32,42])-1
  sigNdz2=np.sum(sum_onebody(obdm,orb1,orb1))

  orb1=np.array([10,20,30,40])-1
  sigNdpi=np.sum(sum_onebody(obdm,orb1,orb1))
  
  #2-body terms
  orb=np.array([14,24,34,44])-1
  sigU=np.sum(sum_U(tbdm,orb))

  orb1=np.array([14,14,24,24,34,34,44,44])-1
  orb2=np.array([24,34,14,44,11,44,24,34])-1
  sigJ=np.sum(sum_J(tbdm,orb1,orb2))

  d=pd.DataFrame({'energy':e,'sigTd':sigTd,'sigT':sigT,'sigNdz':sigNdz,'sigNdpi':sigNdpi,'sigNpz':sigNpz,'sigNdz2':sigNdz2,
  'sigN4s':sigN4s,'sigN2s':sigN2s,'sigNps':sigNps,'sigNpp':sigNpp,'sigNd':sigNd,'sigU':sigU,'sigJ':sigJ,
  'Sz':totspin[run]},index=[0])
  if(df is None): df=d
  else: df=pd.concat((df,d),axis=0)

print(df)
df.to_pickle('scf_gosling.pickle')
