#Generate excitations, energies and rdms for excitations built on CHK state
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from crystal2pyscf import crystal2pyscf_cell
from methods import genex
from basis import basis, minbasis, basis_order
import seaborn as sns
sns.set_style("white")
from pyscf import lo
from functools import reduce 
from pyscf2qwalk import print_qwalk_pbc
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import OrthogonalMatchingPursuit

def calcIAO(cell,mf,basis,occ):
  ''' 
  input: 
  cell and scf (PBC SCF) objects from pyscf and basis
  to calculate IAOs on
  occ is MOs which IAOs should span
  output:
  Calculates 1RDM on orthog atomic orbitals, orthogonalized
  using Lowdin S^1/2
  Returns coefficient matrix for IAOs 
  '''
  s=mf.get_ovlp()[0]

  mo_occ = mf.mo_coeff[0][0][:,occ[0]]
  mo_occ2 = mf.mo_coeff[1][0][:,occ[1]]
  mo_occ=np.concatenate((mo_occ,mo_occ2),axis=1)
  a = lo.iao.iao(cell, mo_occ, minao=basis)
  a = lo.vec_lowdin(a, s)
 
  return a

def rdmIAO(mf,a,occ):
  '''
  input:
  mf object for calculation
  output:
  1rdm for spin up and down
  '''
  s=mf.get_ovlp()[0]
  ind = np.nonzero(mf.mo_occ[0][0][occ[0]]*occ[0])
  mo_occ = mf.mo_coeff[0][0][:,occ[0][ind]]
  mo_occ = reduce(np.dot, (a.T, s, mo_occ))
  dm_u = np.dot(mo_occ, mo_occ.T)
  ind = np.nonzero(mf.mo_occ[1][0][occ[1]]*occ[1])
  mo_occ = mf.mo_coeff[1][0][:,occ[1][ind]]
  mo_occ = reduce(np.dot, (a.T, s, mo_occ))
  dm_d = np.dot(mo_occ, mo_occ.T)
  return np.array([dm_u,dm_d])

#Labels
def makelabels():
  siglist=["2px_1","2px_2","2px_3","2px_4","2py_5","2py_6","2py_7","2py_8"]
  pilist= ["2py_1","2py_2","2py_3","2py_4","2px_5","2px_6","2px_7","2px_8"]
  culabels=["3s","4s","3px","3py","3pz",'3dxy','3dyz','3dz2','3dxz','3dx2y2']
  oxlabels=["2s","2px","2py","2pz"]
  orig_labels=[]
  for i in range(4):
    orig_labels+=["c"+x+"_"+str(i+1) for x in culabels]
  for i in range(8):
    orig_labels+=["o"+x+"_"+str(i+1) for x in oxlabels]
  labels=orig_labels

  my_labels=[]
  for i in range(len(labels)):
    if(labels[i][1:] in siglist): my_labels.append('o2psig_'+labels[i].split("_")[1])
    elif(labels[i][1:] in pilist): my_labels.append('o2ppi_'+labels[i].split("_")[1])
    else: my_labels.append(labels[i])
  labels=my_labels[:]
  return labels

#Parameter sums
def getu(sigu_list):
  sigu=np.zeros((sigu_list.shape[0],sigu_list.shape[1]))
  sigulabels=[]

  labels=makelabels()
  labels=np.array([x.split("_")[0] for x in labels])
  u,indices=np.unique(labels,return_inverse=True)
  
  j=0
  for un in u:
    i=np.where(labels==un)
    sigu[:,j]=np.sum(sigu_list[:,i],axis=2)[:,0]
    sigulabels.append(un+"_u")
    j+=1

  sigulabels=np.array(sigulabels)
  return sigu[:,:sigulabels.shape[0]],sigulabels

def gett(dm,min_dist=0.0,max_dist=4.5):
  #SIGN STRUCTURE ONLY CORRECT FOR NEAREST NEIGHBOR
  #I.E. MAX_DIST=0.25
  labels=makelabels()
  labels2=makelabels()
  labels=np.array([x.split("_")[0] for x in labels])
  u,indices=np.unique(labels,return_inverse=True)

  coords={
    "c_1":[1,1],
    "c_2":[1,0],
    "c_3":[0,1],
    "c_4":[0,0],
    "o_1":[-0.5,1],
    "o_2":[-0.5,0],
    "o_3":[0.5,1],
    "o_4":[0.5,0],
    "o_5":[1,-0.5],
    "o_6":[1,0.5],
    "o_7":[0,-0.5],
    "o_8":[0,0.5]
  } #Coordinates of each atom

  #What to consider identical
  hoplabels=[]
  indices=[]
  for i in range(len(u)):
    for j in range(i,len(u)):
        i1=np.where(labels==u[i])[0]
        i2=np.where(labels==u[j])[0]
        z=0
        for m in i1:
          for n in i2:
            coord1=coords[labels2[m][0]+"_"+labels2[m][-1]]
            coord2=coords[labels2[n][0]+"_"+labels2[n][-1]]
            #dist=np.mod(abs(coord1[0]-coord2[0]),2)**2 + np.mod(abs(coord1[1]-coord2[1]),2)**2
            #dist=np.mod(abs(coord1[0]-coord2[0]),2)**2 + np.mod(abs(coord1[1]-coord2[1]),2)**2
            q1=min(abs(coord1[0]-coord2[0]),abs(abs(coord1[0]-coord2[0])-2))
            q2=min(abs(coord1[1]-coord2[1]),abs(abs(coord1[1]-coord2[1])-2))
            dist=q1**2+q2**2
            assert(dist<=4.5)
            assert(dist>=0.0)
            if((dist<=max_dist) and (dist>=min_dist)):
              dist_string="{0:4.2f}".format(dist)
              hoplabels.append(u[i]+"-"+u[j]+"-"+dist_string)
              indices.append([m,n])
  hoplabels=np.array(hoplabels)
  indices=np.array(indices)
 
  #Construct all hopping sums
  sigt=np.zeros((dm.shape[0],hoplabels.shape[0]))
  sigtlabels=[]
  u=np.unique(hoplabels)
  j=0
  for un in u:
    i=np.where(hoplabels==un)[0]
   
    #Number parameters
    if(un.split("-")[2]=="0.00"):
      sigt[:,j]=np.sum(dm[:,0,indices[i,0],indices[i,1]],axis=1)+\
        np.sum(dm[:,1,indices[i,0],indices[i,1]],axis=1)
      sigtlabels.append(un)
    #NN hopping parameters
    elif(un.split("-")[2]=="0.25"):
      one=un.split("-")[0]
      two=un.split("-")[1]
      #Sign sequence: [R,L,U,D, R,L,D,U, L,R,U,D L,R,D,U]
      if(("c3dx2y2" in one) and ("2psig" in two)): sgn=[1,-1,-1,1, 1,-1,1,-1, -1,1,-1,1, -1,1,1,-1]
      elif(("c3dx2y2" in one) and ("2s" in two)):  sgn=[1,1,-1,-1, 1,1,-1,-1, 1,1,-1,-1, 1,1,-1,-1]
      elif((("c3dz2" in one) or ("c4s" in one)) and ("2psig" in two)): sgn=[1,-1,1,-1, 1,-1,-1,1, -1,1,1,-1, -1,1,-1,1]
      elif((("c3dz2" in one) or ("c4s" in one)) and ("2s" in two)):    sgn=[1,1,1,1,   1,1,1,1,   1,1,1,1,   1,1,1,1]
      else: sgn=np.zeros(16)
      sigt[:,j]=np.dot(dm[:,0,indices[i,0],indices[i,1]],sgn)+\
                np.dot(dm[:,1,indices[i,0],indices[i,1]],sgn)
      sigtlabels.append(un)
    else: 
      print("Don't have hopping for distance "+un.split("_")[2])
      pass
    j+=1

  sigtlabels=np.array(sigtlabels)
  sigt=np.array(sigt)
  return sigt[:,:sigtlabels.shape[0]],sigtlabels

###########################################################################################
#Build IAO 
f="FULLiao.pickle"
a=np.load(f)
print("IAOs built from "+f)

###########################################################################################
#Sample States
#Calculation parameters
act_mo=[np.arange(67,73)-1,np.arange(66,73)-1]
ncore=[66,65]
nact=[1,1]
N=50
Ndet=2
c=0.0
detgen='sd'

direclist=["../FLP_ns"] #COL2, FM, FLP3
E=[-9.2092669022287E+02,-9.2092063265454E+02,-9.2091969201655E+02,-9.2091264054414E+02,-9.2091219109429E+02]
e_list=[]
dm_list=[]
iao_dm_list=[]
sigu_list=[]
zz=0
for direc in direclist:
  cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=direc+"/GRED.DAT",kred=direc+"/KRED.DAT",totspin=ncore[0]+nact[0]-ncore[1]-nact[1])
  e,dm,iao_dm,sigu=genex(mf,a,ncore,nact,act_mo,N,Ndet,detgen,c)
  e_list.append(e-e[0]+E[zz])
  dm_list.append(dm)
  iao_dm_list.append(iao_dm)
  sigu_list.append(sigu)
  zz+=1
  print("Finished excitations for "+direc)
e_list=np.array([j for i in e_list for j in i])
dm_list=np.array([j for i in dm_list for j in i])
iao_dm_list=np.array([j for i in iao_dm_list for j in i])
sigu_list=np.array([j for i in sigu_list for j in i])

#Double occupancy analysis
sigu,sigulabels=getu(sigu_list)

#Variation
'''
for i in range(n.shape[0]):
  plt.plot(sigu[i,:],'.')
plt.xticks(np.arange(sigulabels.shape[0]),sigulabels,rotation=90)
plt.show()
'''

#Hopping analysis
#sigT,sigTlabels=gett(iao_dm_list,min_dist=0.25,max_dist=0.25)
sigT,sigTlabels=gett(iao_dm_list,min_dist=0,max_dist=0.25)

#Variation
'''
for i in range(sigT.shape[0]):
  plt.plot(sigT[i,:],'.')
plt.xticks(np.arange(sigTlabels.shape[0]),sigTlabels,rotation=90)
plt.show()
'''

#Pairplot
data=np.concatenate((e_list[:,np.newaxis]*27.2114,sigu,sigT),axis=1)
df=pd.DataFrame(data,columns=["E"]+list(sigulabels)+list(sigTlabels))

#LinReg
y=df["E"]
X=df.drop(columns=["E"])
ind=[]
for x in list(df):
  #if(("4s" in x) or ("3dz2" in x) or ("3s" in x) or ("3p" in x) or ("3dxy" in x) or ("3dxz" in x) or ("3dyz" in x) or ("2pz" in x) or ("2ppi" in x)): ind.append(x)
  if(("3s" in x) or ("3p" in x) or ("3dxy" in x) or ("3dxz" in x) or ("3dyz" in x) or ("2pz" in x) or ("2ppi" in x)): ind.append(x)
X=X.drop(columns=ind)
#X=X.drop(columns=["o2psig-o2s-0.00"])
X=sm.add_constant(X)
model=sm.OLS(y,X).fit()
print(model.summary())

plt.plot(y,model.predict(X),'og')
plt.plot(y,y,'-')
plt.show()

'''
#Rank checking
u,s,v=np.linalg.svd(X)
print(s)
print(X.shape)
print(np.linalg.matrix_rank(X,tol=1e-6))
'''

#OMP
'''
cscores=[]
cscores_err=[]
scores=[]
conds=[]
nparms=[]
for i in range(1,X.shape[1]+1):
  print("n_nonzero_coefs="+str(i))
  omp = OrthogonalMatchingPursuit(n_nonzero_coefs=i)
  omp.fit(X,y)
  nparms.append(i)
  scores.append(omp.score(X,y))
  tmp=cross_val_score(omp,X,y,cv=5)
  cscores.append(tmp.mean())
  cscores_err.append(tmp.std()*2)
  print("R2: ",omp.score(X,y))
  print("R2CV: ",tmp.mean(),"(",tmp.std()*2,")")
  ind=np.abs(omp.coef_)>0
  Xr=X.values[:,ind]
  conds.append(np.linalg.cond(Xr))
  print("Cond: ",np.linalg.cond(Xr))
  print(np.array(list(X))[ind])
  print(omp.coef_[ind])
'''  
'''
  plt.title(fname)
  plt.xlabel("Predicted energy (eV)")
  plt.ylabel("DFT Energy (eV)")
  plt.plot(omp.predict(X),y,'og')
  plt.plot(y,y,'b-')
  #plt.savefig(fname.split("p")[0][:-1]+".fit_fix.pdf",bbox_inches='tight')
  plt.plot(np.arange(len(omp.coef_)),omp.coef_,'o-',label="Nparms= "+str(i))
'''
'''
plt.axhline(0,color='k',linestyle='--')
plt.xticks(np.arange(len(list(X))),list(X),rotation=90)
plt.title(fname)
plt.ylabel("Parameter (eV)")
plt.legend(loc='best')
plt.savefig(fname.split("p")[0][:-1]+".omp_fix.pdf",bbox_inches='tight')
'''
