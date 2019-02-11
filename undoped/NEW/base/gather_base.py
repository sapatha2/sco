import json 
import pandas as pd
import numpy as np 
import sys
sys.path.append('../../../downfolding/')
from shivesh_downfold_tools import get_qwalk_dm, sum_onebody, sum_J, sum_U, sum_V

def gather_base():
  ''' 
  Gathers all your data and stores into 
  '''
  df=None
  for state in ['chk','col','chkp']:
    f=state+'.vmc_tbdm.gosling.json' 
    print(f)

    data=json.load(open(f,'r'))
    __,__,tbdm,__=get_qwalk_dm(data['properties']['tbdm_basis'])
    obdm,__=get_qwalk_dm(data['properties']['tbdm_basis2'])
    energy=data['properties']['total_energy']['value'][0]*27.2114
    energy_err=data['properties']['total_energy']['error'][0]*27.2114

    orb=np.array([0,1,2,3])
    sigU=np.sum(sum_U(tbdm,orb))
   
    orb1=np.array([0,0,1,1,2,2,3,3])
    orb2=np.array([1,2,0,3,0,3,1,2])
    sigJ=np.sum(sum_J(tbdm,orb1,orb2))

    orb1=np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3])
    orb2=np.array([4,8,6,9,5,9,7,8,6,10,4,11,7,11,5,10])
    sign=np.array([-1,1,1,-1]*4)
    sigT=sum_onebody(obdm,orb1,orb2)
    sigT=np.dot(sign,sigT)

    dat=np.array([energy,energy_err,sigT,sigU,sigJ])
    d=pd.DataFrame(dat[:,np.newaxis].T,columns=['energy','energy_err','sigT','sigU','sigJ'])
    d=d.astype('double')
    d['state']=state
    if(df is None): df=d
    else: df=pd.concat((df,d),axis=0)      
  fout='base_gosling.pickle'
  df.to_pickle(fout)
  return df

import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns 
if __name__=='__main__':
  gather_base()