#Generate excitations, energies and rdms for excitations built on CHK state
import numpy as np 
from functools import reduce

def genex(mf,a,ncore,act,nact,N,Ndet,detgen,c):
  e_list=[]
  dm_list=[]
  iao_dm_list=[]
  assert(Ndet>1)
  assert(N>0)
  assert(c<=1.0)
  assert(c>=0.0)
  s=mf.get_ovlp()[0]
  M0=reduce(np.dot,(a.T, s, mf.mo_coeff[0][0])) #IAO -> MO for spin up
  M1=reduce(np.dot,(a.T, s, mf.mo_coeff[1][0])) #IAO -> MO for spin down 
  
  #Loop states to calculate N+1 states of Ndet+1 determinants
  #First state is always just GS, next are added to GS
  for n in range(N+1):
    #Generate weight object [ CAN BE CHANGED, USING THIS FOR NOWM ... ]
    if(n==0):
      w=np.zeros(Ndet)
      w[0]=1
    else:
      gauss=np.random.normal(size=Ndet-1)
      gauss/=np.sqrt(np.dot(gauss,gauss))
      w=np.zeros(Ndet)+np.sqrt(c)
      w[1:]=gauss*np.sqrt(1-c)

    #Create det_list object [ CAN BE CHANGED FOR SINGLES, DOUBLES, ... ] 
    det_list=np.zeros((Ndet,2,mf.mo_occ.shape[2]))
    det_list[0]=mf.mo_occ[:,0,:]
    for i in range(1,Ndet): 
      det_list[i,:,:ncore]=1
      
      #Generate determinant through all active space (Very fast)
      if(detgen=='a'):
        det_list[i,0,np.random.choice(act[0],size=nact[0],replace=False)]=1
        det_list[i,1,np.random.choice(act[1],size=nact[1],replace=False)]=1
      #Singles excitations only (A bit slow)
      elif(detgen=='s'):
        det_list[i,:,:]=mf.mo_occ[:,0,:]
        spin=np.random.randint(2)
        q=np.random.randint(low=ncore,high=ncore+nact[spin])
        r=np.random.randint(low=ncore+nact[spin],high=act[spin][-1])
        det_list[i,spin,q]=0
        det_list[i,spin,r]=1
      else: 
        print(detgen+" not implemented yet")
        exit(0)

    #Calculate energy 
    el_v=np.einsum('ijk,jk->i',det_list,mf.mo_energy[:,0,:])
    el=np.dot(el_v,w**2)

    #Calculate 1rdm on MO basis 
    dl=np.zeros((det_list.shape[1],det_list.shape[2],det_list.shape[2]))
    dl_v=np.einsum('ijk,i->jk',det_list,w**2)
    dl[0]=np.diag(dl_v[0])
    dl[1]=np.diag(dl_v[1])
    
    offd=np.einsum('ikl,jkl->kij',det_list,det_list)
    for s in [0,1]:
      for a in range(Ndet):
        for b in range(a,Ndet):
          if(offd[s,a,b]==(ncore+nact[s]-1)): #Check for singles excitation
            ind=np.where((det_list[a,s,:]-det_list[b,s,:])!=0)[0]
            M=np.zeros(dl[s].shape)
            M[ind[0],ind[1]]=1
            M[ind[1],ind[0]]=1
            dl[s]+=w[a]*w[b]*M
    
    #Rotate to IAO basis
    rdl=np.zeros((2,M0.shape[0],M0.shape[0]))
    rdl[0]=reduce(np.dot,(M0,dl[0],M0.T))
    rdl[1]=reduce(np.dot,(M1,dl[1],M1.T))

    #Append data to list
    e_list.append(el)
    dm_list.append(dl)
    iao_dm_list.append(rdl)

  e_list=np.array(e_list)
  dm_list=np.array(dm_list)
  iao_dm_list=np.array(iao_dm_list)
  return e_list,dm_list,iao_dm_list
