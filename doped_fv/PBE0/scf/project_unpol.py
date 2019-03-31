import numpy as np 
import pandas as pd 
from functools import reduce 
import matplotlib.pyplot as plt

zz=0
base_coeff=pd.read_pickle('pickles/UNPOL_mo_coeff_g.pickle')
s=pd.read_pickle('pickles/UNPOL_s_g.pickle')
comp = ['FLP','COL','COL2','FM']
for comp_f in comp:
  comp_coeff=pd.read_pickle('pickles/'+comp_f+'_mo_coeff_g.pickle')
  M=np.einsum('ij,sjk->sik',s,base_coeff)
  comp_matrix=np.einsum('sji,sjk->sik',comp_coeff,M) #[s, pol, unpol]
 
  '''
  #No block plots
  plt.matshow(comp_matrix[0,:72,:72],vmax=2,vmin=-2,cmap=plt.cm.bwr)
  plt.xlabel('Polarized')
  plt.ylabel('Unpolarized')
  plt.title('Up channel '+comp_f)
  plt.savefig('plots/'+comp_f+'_s0.pdf',bbox_inches='tight')
  plt.close()

  plt.matshow(comp_matrix[1,:72,:72],vmax=2,vmin=-2,cmap=plt.cm.bwr)
  plt.xlabel('Polarized')
  plt.ylabel('Unpolarized')
  plt.title('Down channel '+comp_f)
  plt.savefig('plots/'+comp_f+'_s1.pdf',bbox_inches='tight')
  plt.close()
  ''' 
  
  #Active polarized MOs
  zz+=1
  act=np.arange(66,73)-1
  if(comp_f=='FM'): act=np.arange(65,73)-1
  plt.subplot(140+zz)
  plt.matshow(comp_matrix[1,act,:72].T,vmax=2,vmin=-2,cmap=plt.cm.bwr,fignum=False)
  plt.ylabel('Polarized')
  plt.xlabel('Unpolarized')
  plt.title('Dn spin '+comp_f)
plt.savefig('plots/act_pol_s1.pdf',bbox_inches='tight')
plt.close()
