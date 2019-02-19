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
 
  for file in [
  'b1/gsw1e-06','b1/gsw0.25','b1/gsw0.5','b1/gsw0.75','b1/gsw1.0',
  'b1/gsw-0.25','b1/gsw-0.5','b1/gsw-0.75',
  'b2/gsw0.25','b2/gsw0.5','b2/gsw0.75','b2/gsw1.0',
  'b3/gsw0.25','b3/gsw0.5','b3/gsw0.75','b3/gsw1.0',
  'b4/gsw0.25','b4/gsw0.5','b4/gsw0.75','b4/gsw1.0',
  ]:
    f=file+'.vmc.gosling.json'
    print(f)
    data=json.load(open(f,'r'))
    __,__,tbdm,__=get_qwalk_dm(data['properties']['tbdm_basis1'])
    obdm,__=get_qwalk_dm(data['properties']['tbdm_basis2'])
    obdm2,__=get_qwalk_dm(data['properties']['tbdm_basis3'])
    energy=data['properties']['total_energy']['value'][0]*27.2114
    energy_err=data['properties']['total_energy']['error'][0]*27.2114

    #2-body (needs to be edited!)
    orb=np.array([0,1,2,3])
    def sum_U_diag(tbdm,orb):
      return 0.5*(tbdm[0,1,orb*4] + tbdm[1,0,orb*4]) 
    #sigU=np.sum(sum_U_diag(tbdm,orb))
    sigU=(sum_U_diag(tbdm,orb))
    print(sigU) 
    exit(0)

    #Hopping
    orb1=np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3])
    orb2=np.array([4,8,6,9,5,9,7,8,6,10,4,11,7,11,5,10])
    sign=np.array([-1,1,1,-1]*4)
    sigT=sum_onebody(obdm,orb1,orb2)
    sigT=np.dot(sign,sigT)

    orb1=np.array([0,0,1,1,2,2,3,3])
    orb2=np.array([1,2,0,3,0,3,1,2])
    sigTd=np.sum(sum_onebody(obdm,orb1,orb2))
 
    #Occupations 
    orb1=np.array([0,1,2,3])
    orb2=np.array([0,1,2,3])
    sigNd=np.sum(sum_onebody(obdm,orb1,orb2))
    
    orb1=np.array([4,5,6,7,8,9,10,11])
    orb2=np.array([4,5,6,7,8,9,10,11])
    sigNps=np.sum(sum_onebody(obdm,orb1,orb2))
 
    orb1=np.array([12,13,14,15])
    orb2=np.array([12,13,14,15])
    sigN4s=np.sum(sum_onebody(obdm,orb1,orb2))
 
    orb1=np.arange(8)
    orb2=np.arange(8)
    sigNpp=np.sum(sum_onebody(obdm2,orb1,orb2))

    dat=np.array([energy,energy_err,sigT,sigTd,sigU,sigNd,sigNps,sigN4s,sigNpp])
    d=pd.DataFrame(dat[:,np.newaxis].T,columns=['energy','energy_err','sigT','sigTd','sigU','sigNd','sigNps','sigN4s','sigNpp'])
    d=d.astype('double')
    d['path']=file.split("/")[0][1:]
    if(df is None): df=d
    else: df=pd.concat((df,d),axis=0)   
  fout='b_gosling.pickle'
  df.to_pickle(fout)
  return df

import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns 
if __name__=='__main__':
  df=gather_base()
