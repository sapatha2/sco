#COLLECT T, U, J FOR MY 4 RELEVANT STATES
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
'''
from crystal2pyscf import crystal2pyscf_cell
from pyscf2qwalk import print_qwalk_pbc
from basis import basis, minbasis, basis_order
from pyscf import lo
from functools import reduce 
from downfold_tools import gen_slater_tbdm, sum_onebody, sum_J, sum_U
'''
import seaborn as sns 

'''
a=np.load('iao.pickle')
for direc in ['../../PBE0/CHK','../../PBE0/COL']:
  cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=direc+"/GRED.DAT",kred=direc+"/KRED.DAT",totspin=0)
  s=mf.get_ovlp()[0]

  #Build excitation
  for ex in [0,1]:
    print(direc,ex)

    #Build rdms
    obdm=np.zeros((2,a.shape[1],a.shape[1]))
    for spin in [0,1]:
      mo_to_iao=reduce(np.dot,(a.T, s, mf.mo_coeff[spin][0]))
      mo_rdm=np.diag(mf.mo_occ[spin][0])
      
      if(ex and spin): 
        tmp=mf.mo_occ[spin][0]
        tmp[65]=0
        tmp[66]=1
        mo_rdm=np.diag(tmp)

      obdm[spin]=reduce(np.dot,(mo_to_iao,mo_rdm,mo_to_iao.T))
    tbdm,__=gen_slater_tbdm(obdm)

    orb1=np.array([14,14,14,14,24,24,24,24,34,34,34,34,44,44,44,44])-1
    orb2=np.array([46,63,54,67,50,67,58,63,54,71,46,75,58,75,50,71])-1
    sign=np.array([-1,1,1,-1]*4)
    sigT=sum_onebody(obdm,orb1,orb2)
    sigT=np.dot(sign,sigT)
    print('sigT: ',sigT)

    orb=np.array([14,24,34,44])-1
    sigU=np.sum(sum_U(tbdm,orb))
    print('sigU: ',sigU)

    orb1=np.array([14,14,24,24,34,34,44,44])-1
    orb2=np.array([24,34,14,44,14,44,24,34])-1
    sigJ=np.sum(sum_J(tbdm,orb1,orb2))
    print('sigJ: ',sigJ)
'''

name=['CHK','CHKp','COL','COLp']
U=[1.43,2.11,1.30,2.07]
J=[-0.76,-0.33,-0.02,0.04]
T=[3.91,3.53,3.58,3.33]
df=pd.DataFrame({'U':U,'J':J,'T':T,'state':name})
sns.pairplot(df,vars=['T','U','J'],hue='state',markers=['o','o','o','.'])
#plt.show()
plt.savefig('../plots/base_dft.pdf',bbox_inches='tight')
