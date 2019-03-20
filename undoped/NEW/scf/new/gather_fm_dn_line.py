#Collect single determinant excitations
import numpy as np 
import pandas as pd
import seaborn as sns 
from functools import reduce 
import matplotlib.pyplot as plt
from downfold_tools import sum_onebody

#from crystal2pyscf import crystal2pyscf_cell
#from basis import basis, minbasis, basis_order
#direc='../../../PBE0/FM'
#cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=direc+"/GRED.DAT",kred=direc+"/KRED.DAT",totspin=0)
#mf.mo_coeff.dump('fm_mocoeff.pickle')
#mf.get_ovlp().dump('s.pickle')
#mf.mo_energy.dump('fm_moenergy.pickle')
#print(mf.mo_energy.shape)
#exit(0)

#Builds mo rdms for a two determinant state
#with given ground state weight
def mo_rdm_fm(mo_occ1,mo_occ2,w):
  #Preliminary load ins
  a=np.load('iao.pickle')
  fm_mocoeff=np.load('fm_mocoeff.pickle')[:,0,:,:]
  s=np.load('s.pickle')
  
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
  
  d=pd.DataFrame({'energy':e*27.2114,'sigTd':sigTd,'sigT':sigT,'sigNdz':sigNdz,'sigNdpi':sigNdpi,'sigNpz':sigNpz,'sigNdz2':sigNdz2,
  'sigN4s':sigN4s,'sigN2s':sigN2s,'sigNps':sigNps,'sigNpp':sigNpp,'sigNd':sigNd,'sigU':sigU},index=[0])
  return d 

#Generate sigU for a sum of two determinants 
def get_U_fm(mo_occ1,mo_occ2,w,M0,M1):
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
def gather_line_fm(rem,add,gsws):
  #Preliminary load ins
  a=np.load('iao.pickle')
  fm_mocoeff=np.load('fm_mocoeff.pickle')[:,0,:,:]
  fm_moenergy=np.load('fm_moenergy.pickle')[:,0,:]
  s=np.load('s.pickle')
  M0=reduce(np.dot,(a.T, s, fm_mocoeff[0])) 
  M1=reduce(np.dot,(a.T, s, fm_mocoeff[1])) 

  df=None
  for gsw in gsws:
    #Get MO RDM
    w=np.array([np.sqrt(gsw),np.sqrt(1-gsw)])
    mo_occ1=np.zeros((fm_mocoeff.shape[:-1]))
    mo_occ1[0,:68]=1
    mo_occ1[1,:64]=1
    mo_occ2=np.zeros((fm_mocoeff.shape[:-1]))
    mo_occ2[0,:68]=1
    mo_occ2[1,:64]=1
    mo_occ2[1,rem]=0
    mo_occ2[1,add]=1
    dl=mo_rdm_fm(mo_occ1,mo_occ2,w)

    #Convert to IAO RDM
    R=np.einsum('ij,jk->ik',dl[0],M0.T)
    dm_u=np.einsum('ij,jk->ik',M0,R)
    R=np.einsum('ij,jk->ik',dl[1],M1.T)
    dm_d=np.einsum('ij,jk->ik',M1,R)
    obdm=np.array([dm_u,dm_d])

    #Generate sigU separately since we no longer have single determinant
    sigU=get_U_fm(mo_occ1,mo_occ2,w,M0,M1)

    #Get energy from eigenvalues 
    e=w[1]**2*(fm_moenergy[1,add]-fm_moenergy[1,rem])

    #Gather df row
    d=get_df_row(obdm,sigU,e)

    if(df is None): df=d
    else: df=pd.concat((df,d),axis=0)
  return df

if __name__=='__main__':
  #Smallest sample set, sigma only, no pi or dz2,4s
  rem_list=list(np.arange(24,64))*8
  add_list=[64]*40+[65]*40+[66]*40+[67]*40+[68]*40+[69]*40+[70]*40+[71]*40
  gsws=np.arange(1.0,-0.1,-0.1)

  full_df=None
  for rem,add in zip(rem_list,add_list):
    print(rem,add)
    df=gather_line_fm(rem,add,gsws)
    df['add']=add
    df['rem']=rem
    if(full_df is None): full_df=df
    else: full_df=pd.concat((full_df,df),axis=0)

  print(full_df)
  print(full_df.shape)
  full_df.to_pickle('fm_dn_line_gosling.pickle')
