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
import itertools 

a=np.load('pickles/iao_g.pickle')
b=np.load('pickles/UNPOL_mo_coeff_g.pickle')[0][:,[55,65,66,67,68,69,70,71]]
df=None

#Generates doubles excitations given 
#a set of excitations (remove, add, spin channel)
def gen_doubles(excit):
  excit_list=np.array(list(itertools.combinations(excit,2)))
  filt=np.ones(excit_list.shape[0],dtype=bool)
  for i in range(len(excit_list)):
    x=excit_list[i]
    if(x[0][2] == x[1][2]):
      filt[i]=False

  excit_list=excit_list[filt]
  return list(excit_list)

#Excitation list for just spin 1 states
excit1 = set([  
(66,66,0),       #Remove, add, spin channel
(66,67,0),
(66,68,0),
(66,69,0),
(66,70,0),
(66,71,0),

(65,65,1),
(65,66,1),
(65,67,1),
(65,68,1),
(65,69,1),
(65,70,1),
(65,71,1)
])
excit_list=gen_doubles(excit1)*3 #FLP, COL, COL2

#Excitation list for just spin 3 states
excit3 = set([  
(67,67,0),
(67,68,0),
(67,69,0),
(67,70,0),
(67,71,0),

(64,64,1),
(64,65,1),
(64,66,1),
(64,67,1),
(64,68,1),
(64,69,1),
(64,70,1),
(64,71,1)
])
excit_list+=gen_doubles(excit3)

#print(np.array(excit_list)[[4,74,116,150]])
#print(np.array(excit_list)[[  0,   1 ,  2 ,  3 ,  5,   6 , 15,  30 , 31,  32 , 33 , 46 , 52,  57,  62,  66,  70,  72,
#  73 , 75,  80 , 81  ,88,  94,  99, 104 ,108, 112, 114, 115, 117, 122 ,123 ,127 ,134, 138,
# 142, 146, 151, 152 ,153 ,154, 155, 156]])
#exit(0)

direcs=['FLP']*42+['COL']*42+['COL2']*42+['FM']*40
base_energy=[0]*42+[0.16]*42+[0.19]*42+[0.38]*40
for run in range(len(excit_list)):
  print(direcs[run],run)
  
  s=pd.read_pickle('pickles/'+direcs[run]+'_s_g.pickle')
  mo_occ=pd.read_pickle('pickles/'+direcs[run]+'_mo_occ_g.pickle')
  mo_coeff=pd.read_pickle('pickles/'+direcs[run]+'_mo_coeff_g.pickle')
  mo_energy=pd.read_pickle('pickles/'+direcs[run]+'_mo_energy_g.pickle')
  
  obdm=np.zeros((2,a.shape[1],a.shape[1]))
  obdm2=np.zeros((2,b.shape[1],b.shape[1]))
  e=base_energy[run]
  
  for excit in excit_list[run]: #Two excitations in this case, always one in spin-up other in spin-down
    rem=excit[0]
    add=excit[1]
    spin=excit[2]
    
    mo_rdm=mo_occ[spin]
    mo_to_iao=reduce(np.dot,(a.T, s, mo_coeff[spin]))
    mo_to_unp=reduce(np.dot,(b.T, s, mo_coeff[spin]))
    
    mo_rdm[rem]=0
    mo_rdm[add]=1
    e+=(mo_energy[spin][add]-mo_energy[spin][rem])*27.2114
     
    obdm[spin]=reduce(np.dot,(mo_to_iao,np.diag(mo_rdm),mo_to_iao.T))
    obdm2[spin]=reduce(np.dot,(mo_to_unp,np.diag(mo_rdm),mo_to_unp.T))
  print(excit_list[run],e)
  tbdm,__=gen_slater_tbdm(obdm)

  #Hopping
  orb1=np.array([14,14,14,14,24,24,24,24,34,34,34,34,44,44,44,44])-1
  orb2=np.array([46,63,54,67,50,67,58,63,54,71,46,75,58,75,50,71])-1
  sign=np.array([-1,1,1,-1]*4)
  sigTdp=sum_onebody(obdm,orb1,orb2)
  sigTdp=2*np.dot(sign,sigTdp)

  orb1=np.array([ 6, 6, 6, 6,16,16,16,16,26,26,26,26,36,36,36,36])-1
  orb2=np.array([46,63,54,67,50,67,58,63,54,71,46,75,58,75,50,71])-1
  sign=np.array([-1,-1,1,1]*4)
  sigTps=sum_onebody(obdm,orb1,orb2)
  sigTps=2*np.dot(sign,sigTps)

  orb1=np.array([ 6,16,26,36])-1
  orb2=np.array([14,24,34,44])-1
  sign=np.array([1,1,1,1])
  sigTds=sum_onebody(obdm,orb1,orb2)
  sigTds=2*np.dot(sign,sigTds)

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

  orb1=np.arange(4)
  sigNsr=np.sum(sum_onebody(obdm,orb1,orb1))

  orb1=np.arange(8)
  sigN_unp=sum_onebody(obdm2,orb1,orb1)

  orb1=[0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,4,4,4,5,5,6]
  orb2=[1,2,3,4,5,6,7,2,3,4,5,6,7,3,4,5,6,7,4,5,6,7,5,6,7,6,7,7]
  sigT_unp=sum_onebody(obdm2,orb1,orb2)
  sigT_unp_lab=['sigT_'+str(orb1[i])+'_'+str(orb2[i]) for i in range(len(orb1))]

  #2-body terms
  orb=np.array([14,24,34,44])-1
  sigUd=np.sum(sum_U(tbdm,orb))

  orb=np.array([46,50,54,58,63,67,71,75])-1
  sigUp=np.sum(sum_U(tbdm,orb))
  
  orb=np.array([6,16,26,36])-1
  sigUs=np.sum(sum_U(tbdm,orb))

  orb1=np.array([14,14,24,24,34,34,44,44])-1
  orb2=np.array([24,34,14,44,11,44,24,34])-1
  sigJd=np.sum(sum_J(tbdm,orb1,orb2))

  data={'energy':e,
  'sigTds':sigTds,'sigTdp':sigTdp,'sigTps':sigTps,
  'Nd':sigNd,'Np':sigNps,'Ns':sigN4s,'Nsr':sigNsr,
  'sigUd':sigUd,'sigUp':sigUp,'sigUs':sigUs,'sigJd':sigJd,
  'N2s':sigN2s,'Ndz':sigNdz,'Ndpi':sigNdpi,'Ndz2':sigNdz2,'Npz':sigNpz,'Npp':sigNpp,
  'basestate':direcs[run]}
  for i in range(len(sigN_unp)):
    data['N'+str(i)]=sigN_unp[i]
  for i in range(len(sigT_unp)):
    data[sigT_unp_lab[i]]=sigT_unp[i]

  d=pd.DataFrame(data,index=[run])

  if(df is None): df=d
  else: df=pd.concat((df,d),axis=0)

print(df)
df.to_pickle('pickles/full_sd_gosling.pickle')
