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
  'p1/gsw0.25','p1/gsw0.5','p1/gsw0.75','p1/gsw1.0',
  'p2/gsw0.25','p2/gsw0.5','p2/gsw0.75','p2/gsw1.0',
  'p2/gsw-0.25','p2/gsw-0.5','p2/gsw-0.75',
  'p3/gsw0.25','p3/gsw0.5','p3/gsw0.75','p3/gsw1.0',
  'p3/gsw-0.25','p3/gsw-0.5','p3/gsw-0.75',
  'p4/gsw0.25','p4/gsw0.5','p4/gsw0.75','p4/gsw1.0',
  'p5/gsw0.25','p5/gsw0.5','p5/gsw0.75','p5/gsw1.0',
  'p6/gsw0.25','p6/gsw0.5','p6/gsw0.75','p6/gsw1.0',
  ]:
    if("base" in file): f=file+'.vmc_tbdm.gosling.json'
    else: f=file+'.vmc.gosling.json'
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

    orb1=np.array([0,0,1,1,2,2,3,3])
    orb2=np.array([1,2,0,3,0,3,1,2])
    sigTd=np.sum(sum_onebody(obdm,orb1,orb2))
  
    orb1=np.array([0,1,2,3])
    orb2=np.array([0,1,2,3])
    sigNd=np.sum(sum_onebody(obdm,orb1,orb2))
    
    orb1=np.array([4,5,6,7,8,9,10,11])
    orb2=np.array([4,5,6,7,8,9,10,11])
    sigNps=np.sum(sum_onebody(obdm,orb1,orb2))

    dat=np.array([energy,energy_err,sigT,sigTd,sigU,sigJ,sigNd,sigNps])
    d=pd.DataFrame(dat[:,np.newaxis].T,columns=['energy','energy_err','sigT','sigTd','sigU','sigJ','sigNd','sigNps'])
    d=d.astype('double')
    d['path']=file[1]
    if('chkp' in file): d['path']='chkp'
    elif('col' in file): d['path']='col'
    elif('chk' in file): d['path']='chk'
    if('base' in f): d['gsw']='1'
    else: d['gsw']=f.split(".")[0]+f.split(".")[1]
    if(df is None): df=d
    else: df=pd.concat((df,d),axis=0)   
  fout='p_gosling.pickle'
  df.to_pickle(fout)
  return df

import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns 
if __name__=='__main__':
  df=gather_base()
