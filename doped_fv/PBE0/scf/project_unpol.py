import numpy as np 
import pandas as pd 
from functools import reduce 
import matplotlib.pyplot as plt


#Project active pol onto unpolarized
zz=0
base_coeff=pd.read_pickle('pickles/UNPOL_mo_coeff_g.pickle')
s=pd.read_pickle('pickles/FLP_s_g.pickle')
comp = ['FLP','COL','COL2','FM']
for spin in [0,1]:
  zz=0
  for comp_f in comp:
    zz+=1
    comp_coeff=pd.read_pickle('pickles/'+comp_f+'_mo_coeff_g.pickle')
    M=np.einsum('ij,sjk->sik',s,base_coeff)
    comp_matrix=np.einsum('sji,sjk->sik',comp_coeff,M) #[s, pol, FLP]

    if(spin==0):
      act=np.arange(67,73)-1
      if(comp_f=='FM'): act=np.arange(68,73)-1
    else:
      act=np.arange(66,73)-1
      if(comp_f=='FM'): act=np.arange(65,73)-1
    plt.subplot(410+zz)
    plt.matshow(comp_matrix[spin,act,:],vmax=1,vmin=-1,cmap=plt.cm.bwr,fignum=False)
    plt.ylabel(comp_f)
    plt.xlabel('Unpol')
  plt.suptitle('Active bands S='+str(spin))
  #plt.show()
  plt.savefig('plots/act_unpol_s'+str(spin)+'_full.pdf',bbox_inches='tight')

'''
#Project active unpol to polarized
zz=0
base_coeff=pd.read_pickle('pickles/UNPOL_mo_coeff_g.pickle')
s=pd.read_pickle('pickles/UNPOL_s_g.pickle')
comp = ['FLP','COL','COL2','FM']
for spin in [0,1]:
  zz=0
  for comp_f in comp:
    zz+=1
    comp_coeff=pd.read_pickle('pickles/'+comp_f+'_mo_coeff_g.pickle')
    M=np.einsum('ij,sjk->sik',s,base_coeff)
    comp_matrix=np.einsum('sji,sjk->sik',comp_coeff,M) #[s, pol, FLP]

    act=[55,65,66,67,68,69,70,71]
    plt.subplot(410+zz)
    plt.matshow(comp_matrix[spin,:72,act],vmax=1,vmin=-1,cmap=plt.cm.bwr,fignum=False)
    plt.ylabel('Unpol')
    plt.xlabel(comp_f)
  plt.suptitle('Active bands S='+str(spin))
  plt.savefig('plots/act_pol_s'+str(spin)+'.pdf',bbox_inches='tight')
'''
