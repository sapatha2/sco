#Collect single determinant excitations
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

a=np.load('pickles/UNPOL_mo_coeff_g.pickle')[0]
a=a[:,:72]
b=np.load('pickles/iao_g.pickle')

df=None
basestate=[0]*12+[1]*12+[2]*12+[3]*12
direcs=['FLP']*12+['COL']*12+['COL2']*12+['FM']*12
base_energy=[0]*12+[0.16]*12+[0.19]*12+[0.38]*12
rem=[[[],[]],
     [[66],[]],
     [[66],[]],
     [[66],[]],
     [[66],[]],
     [[66],[]],
     [[],[65]],
     [[],[65]],
     [[],[65]],
     [[],[65]],
     [[],[65]],
     [[],[65]],
     [[],[]],
     [[66],[]],
     [[66],[]],
     [[66],[]],
     [[66],[]],
     [[66],[]],
     [[],[65]],
     [[],[65]],
     [[],[65]],
     [[],[65]],
     [[],[65]],
     [[],[65]],
     [[],[]],
     [[66],[]],
     [[66],[]],
     [[66],[]],
     [[66],[]],
     [[66],[]],
     [[],[65]],
     [[],[65]],
     [[],[65]],
     [[],[65]],
     [[],[65]],
     [[],[65]],
     [[],[]],
     [[67],[]],
     [[67],[]],
     [[67],[]],
     [[67],[]],
     [[],[64]],
     [[],[64]],
     [[],[64]],
     [[],[64]],
     [[],[64]],
     [[],[64]],
     [[],[64]]]
add=[[[],[]],
     [[67],[]],
     [[68],[]],
     [[69],[]],
     [[70],[]],
     [[71],[]],
     [[],[66]],
     [[],[67]],
     [[],[68]],
     [[],[69]],
     [[],[70]],
     [[],[71]],
     [[],[]], 
     [[67],[]],
     [[68],[]],
     [[69],[]],
     [[70],[]],
     [[71],[]],
     [[],[66]],
     [[],[67]],
     [[],[68]],
     [[],[69]],
     [[],[70]],
     [[],[71]],
     [[],[]], 
     [[67],[]],
     [[68],[]],
     [[69],[]],
     [[70],[]],
     [[71],[]],
     [[],[66]],
     [[],[67]],
     [[],[68]],
     [[],[69]],
     [[],[70]],
     [[],[71]],
     [[],[]], 
     [[68],[]],
     [[69],[]],
     [[70],[]],
     [[71],[]],
     [[],[65]],
     [[],[66]],
     [[],[67]],
     [[],[68]],
     [[],[69]],
     [[],[70]],
     [[],[71]]]
for run in range(len(direcs)):
  print(direcs[run],run)
  s=pd.read_pickle('pickles/'+direcs[run]+'_s_g.pickle')
  mo_occ=pd.read_pickle('pickles/'+direcs[run]+'_mo_occ_g.pickle')
  mo_coeff=pd.read_pickle('pickles/'+direcs[run]+'_mo_coeff_g.pickle')
  mo_energy=pd.read_pickle('pickles/'+direcs[run]+'_mo_energy_g.pickle')
  obdm=np.zeros((2,a.shape[1],a.shape[1]))
  e=base_energy[run]
  
  for spin in [0,1]:
    mo_rdm=mo_occ[spin]
    mo_to_iao=reduce(np.dot,(a.T, s, mo_coeff[spin]))
    if(len(rem[run][spin])>0):
      mo_rdm[rem[run][spin]]=0
      mo_rdm[add[run][spin]]=1
    obdm[spin]=reduce(np.dot,(mo_to_iao,np.diag(mo_rdm),mo_to_iao.T))
  
  obdm2=np.zeros((2,b.shape[1],b.shape[1]))
  for spin in [0,1]:
    mo_rdm=mo_occ[spin]
    mo_to_iao=reduce(np.dot,(b.T, s, mo_coeff[spin]))
    if(len(rem[run][spin])>0):
      mo_rdm[rem[run][spin]]=0
      mo_rdm[add[run][spin]]=1
      e+=(mo_energy[spin][add[run][spin]]-mo_energy[spin][rem[run][spin]])[0]*27.2114
    obdm2[spin]=reduce(np.dot,(mo_to_iao,np.diag(mo_rdm),mo_to_iao.T))
  tbdm,__=gen_slater_tbdm(obdm2)

  #1-body terms
  #orb1=np.array([0,1,2,3,4,5,6,7,0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,4,4,4,5,5,6])
  #orb2=np.array([0,1,2,3,4,5,6,7,1,2,3,4,5,6,7,2,3,4,5,6,7,3,4,5,6,7,4,5,6,7,5,6,7,6,7,7])
  orb1=np.arange(72)
  orb2=orb1
  sigN=sum_onebody(obdm,orb1,orb2)
  sigN_labels=['sigN_'+str(orb1[i])+'_'+str(orb2[i]) for i in range(len(orb1))]
  #print("Mo Tr: "+str(sum(sigN)))

  obdm=obdm[:,[55,65,66,67,68,69,70,71],:]
  obdm=obdm[:,:,[55,65,66,67,68,69,70,71]]
  orb1=np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,4,4,4,5,5,6])
  orb2=np.array([1,2,3,4,5,6,7,2,3,4,5,6,7,3,4,5,6,7,4,5,6,7,5,6,7,6,7,7])
  sigT=sum_onebody(obdm,orb1,orb2)
  sigT_labels=['sigT_'+str(orb1[i])+'_'+str(orb2[i]) for i in range(len(orb1))]

  #2-body terms
  orb=np.array([14,24,34,44])-1
  sigU=np.sum(sum_U(tbdm,orb))

  orb1=np.array([14,14,24,24,34,34,44,44])-1
  orb2=np.array([24,34,14,44,11,44,24,34])-1
  sigJ=np.sum(sum_J(tbdm,orb1,orb2))

  orb=np.array([46,50,54,58,63,67,71,75])-1
  sigUp=np.sum(sum_U(tbdm,orb))

  data=np.array([e]+list(sigN)+list(sigT)+[sigU]+[sigUp]+[sigJ]+[basestate[run]])
  d=pd.DataFrame(data[:,np.newaxis].T,columns=['energy']+sigN_labels+sigT_labels+['sigU','sigUp','sigJ']+['basestate'],index=[run])
  #d=pd.DataFrame({'energy':e,'sigTd':sigTd,'sigT':sigT,'sigNdz':sigNdz,'sigNdpi':sigNdpi,'sigNpz':sigNpz,'sigNdz2':sigNdz2,
  #'sigN4s':sigN4s,'sigN2s':sigN2s,'sigNps':sigNps,'sigNpp':sigNpp,'sigNd':sigNd,'sigU':sigU,'sigJ':sigJ,
  #'basestate':direcs[run]},index=[0])
  if(df is None): df=d
  else: df=pd.concat((df,d),axis=0)

print(df)
df.to_pickle('pickles/sd_gosling_mo.pickle')
