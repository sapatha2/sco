#Collect single determinant excitations
import numpy as np 
import pandas as pd
import seaborn as sns 
from functools import reduce 
import matplotlib.pyplot as plt
from downfold_tools import sum_onebody

#Builds mo rdms for a two determinant state
#with given ground state weight
def mo_rdm(mo_occ1,mo_occ2,w):
  #Preliminary load ins
  a=np.load('pickles/iao_g.pickle')
  chk_mocoeff=np.load('pickles/FLP_mo_coeff_g.pickle')
  s=np.load('pickles/FLP_s_g.pickle')
  
  #RDM for each determinant
  mo_dm1=np.einsum('si,ij->sij',mo_occ1,np.eye(mo_occ1.shape[1],mo_occ1.shape[1]))
  mo_dm2=np.einsum('si,ij->sij',mo_occ2,np.eye(mo_occ2.shape[1],mo_occ2.shape[1]))
  mo_dm=np.array([mo_dm1,mo_dm2])
  
  #Calculate RDM sum
  dl=np.zeros(mo_dm.shape[1:])
  Ndet=mo_dm.shape[0]
  for n in range(Ndet):
    dl+=mo_dm[n,:,:,:]*w[n]**2
      
    #Cross terms
    diag0=np.diag(mo_dm[n,0,:,:]-mo_dm[0,0,:,:])
    diag1=np.diag(mo_dm[n,1,:,:]-mo_dm[0,1,:,:])
    i0=np.where(diag0!=0)[0]
    i1=np.where(diag1!=0)[0]
    if(i0.shape[0]>0):
      dl[0,i0[0],i0[1]]+=w[0]*w[n]
      dl[0,i0[1],i0[0]]+=w[0]*w[n]
    if(i1.shape[0]>0):
      dl[1,i1[0],i1[1]]+=w[0]*w[n]
      dl[1,i1[1],i1[0]]+=w[0]*w[n]
  return dl

def get_df_row(obdm,sigU,e):
  #1-body terms
  orb1=np.array([0,1,2,3,4,5,6,7,0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,4,4,4,5,5,6])
  orb2=np.array([0,1,2,3,4,5,6,7,1,2,3,4,5,6,7,2,3,4,5,6,7,3,4,5,6,7,4,5,6,7,5,6,7,6,7,7])
  sigN=sum_onebody(obdm,orb1,orb2)
  sigN_labels=['sigN_'+str(orb1[i])+'_'+str(orb2[i]) for i in range(len(orb1))]
  #print("Mo Tr: "+str(sum(sigN)))

  data=np.array([e]+list(sigN)+[sigU])
  d=pd.DataFrame(data[:,np.newaxis].T,columns=['energy']+sigN_labels+['sigU'],index=[0])
  return d 

#Generate sigU for a sum of two determinants 
def get_U(mo_occ1,mo_occ2,w,M0,M1):
  M=np.array([M0,M1])
  mo_dm1=np.einsum('si,ij->sij',mo_occ1,np.eye(mo_occ1.shape[1],mo_occ1.shape[1]))
  mo_dm2=np.einsum('si,ij->sij',mo_occ2,np.eye(mo_occ2.shape[1],mo_occ2.shape[1]))
  mo_dm=np.array([mo_dm1,mo_dm2])
  
  sigU=[]
  for dl in mo_dm:
    R=np.einsum('ij,jk->ik',dl[0],M0.T)
    dm_u=np.einsum('ij,jk->ik',M0,R)
    R=np.einsum('ij,jk->ik',dl[1],M1.T)
    dm_d=np.einsum('ij,jk->ik',M1,R)
    sigU.append(np.sum(dm_u[[13,23,33,43]]*dm_d[[13,23,33,43]]))

  sigU=np.array(sigU)
  return np.dot(sigU,w**2)

#Singles excitations on checkerboard state
#rem, add are floats, should be what you want to remove and add
#gsws are a list of gsws that you want to calculate the sum with 
def gather_line(rem,add,gsws):
  #Preliminary load ins
  a=np.load('pickles/UNPOL_mo_coeff_g.pickle')[0]
  a=a[:,[55,65,66,67,68,69,70,71]]
  b=np.load('pickles/iao_g.pickle')
  chk_mocoeff=np.load('pickles/FLP_mo_coeff_g.pickle')
  chk_moenergy=np.load('pickles/FLP_mo_energy_g.pickle')
  s=np.load('pickles/FLP_s_g.pickle')
  #pol to unpol
  m0=reduce(np.dot,(a.T, s, chk_mocoeff[0])) 
  m1=reduce(np.dot,(a.T, s, chk_mocoeff[1])) 
  #pol to iao
  M0=reduce(np.dot,(b.T, s, chk_mocoeff[0])) 
  M1=reduce(np.dot,(b.T, s, chk_mocoeff[1])) 

  df=None
  for gsw in gsws:
    #Get MO RDM
    w=np.array([np.sqrt(gsw),np.sqrt(1-gsw)])
    mo_occ1=np.zeros((chk_mocoeff.shape[:-1]))
    mo_occ1[0,:67]=1
    mo_occ1[0,:66]=1
    mo_occ2=np.zeros((chk_mocoeff.shape[:-1]))
    mo_occ2[0,:67]=1
    mo_occ2[0,:66]=1
    mo_occ2[0,rem]=0
    mo_occ2[0,add]=1
    dl=mo_rdm(mo_occ1,mo_occ2,w)

    #Convert to IAO RDM
    R=np.einsum('ij,jk->ik',dl[0],m0.T)
    dm_u=np.einsum('ij,jk->ik',m0,R)
    R=np.einsum('ij,jk->ik',dl[1],m1.T)
    dm_d=np.einsum('ij,jk->ik',m1,R)
    obdm=np.array([dm_u,dm_d])

    #Generate sigU separately since we no longer have single determinant
    sigU=get_U(mo_occ1,mo_occ2,w,M0,M1)

    #Get energy from eigenvalues 
    e=w[1]**2*(chk_moenergy[0,add]-chk_moenergy[0,rem])

    #Gather df row
    d=get_df_row(obdm,sigU,e)

    if(df is None): df=d
    else: df=pd.concat((df,d),axis=0)
  return df

if __name__=='__main__':
  #Smallest sample set, sigma only, no pi or dz2,4s
  rem_list=[66]*5
  add_list=[67,68,69,70,71]
  gsws=np.arange(-1.00,1.10,0.1)
  print(gsws)

  full_df=None
  for rem,add in zip(rem_list,add_list):
    print(rem,add)
    df=gather_line(rem,add,gsws)
    df['add']=add
    df['rem']=rem
    if(full_df is None): full_df=df
    else: full_df=pd.concat((full_df,df),axis=0)

  print(full_df)
  print(full_df.shape)
  full_df.to_pickle('pickles/FLP_up_line_gosling_g.pickle')
