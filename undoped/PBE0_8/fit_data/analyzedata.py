import numpy as np 
import matplotlib.pyplot as plt
import pickle 
import pandas as pd 

with open('gendata.pickle', 'rb') as handle:
  data = pickle.load(handle)

N=len(data['energy'])
#Plot energies
e=np.array(data['energy'])
e-=e[0]
e*=27.2

'''
plt.plot(sorted(e),'o')
plt.ylabel("Energy - CHK, eV")
plt.show()

#Plot nelec of 1rdm
rdms=data['1rdm']
nelec=[np.trace(x[0])+np.trace(x[1]) for x in rdms]
plt.plot(nelec,'o')
plt.ylabel("Tr(up) + Tr(dn)")
plt.xlabel("Excitation")
plt.show()
'''

#Array of copper occupations 
ncu=10
no=4
nsr=1

n=[]
u=[]
for rdms in data['1rdm']:
  tmp=[]
  for s in range(2):
    #Copper part 
    cu=np.diag(rdms[s])[:8*ncu]
    cu=np.split(cu,8)
    cu=np.sum(cu,axis=0)/8.
  
    #Oxygen part
    o=np.diag(rdms[s])[8*ncu:8*ncu+16*no]
    o=np.split(o,16)
    o=np.sum(o,axis=0)/8.
  
    #Sr part
    sr=np.diag(rdms[s])[8*ncu+16*no:8*ncu+16*no+8*nsr]
    sr=np.split(sr,8)
    sr=np.sum(sr,axis=0)/8.
    
    tmp.append(np.concatenate((cu,o,sr),axis=None))
  n.append(np.concatenate((tmp[0],tmp[1]),axis=None))

n=np.array(n) #(Cu, O, Sr)up - (Cu, O, Sr)dn

#Array to consider covariance of, numbers and energy
X=np.concatenate((e[:,np.newaxis],n),axis=1)
cols=[
"E",
"3su","4su","3pxu","3pyu","3pzu",'3dxyu','3dyzu','3dz2u','3dxzu','3dx2y2u',
'2su','2pxu','2pyu','2pzu','5du',
"3sd","4sd","3pxd","3pyd","3pzd",'3dxyd','3dyzd','3dz2d','3dxzd','3dx2y2d',
'2sd','2pxd','2pyd','2pzd','5dd',
]
df=pd.DataFrame(data=X,columns=cols)

#Plot variation in numbers and energy
for i in range(1,df.shape[1]):
  plt.plot(i*np.ones(df.shape[0])-1,df.values[:,i],'o')
  plt.xticks(np.arange(len(cols)-1),cols[1:],rotation=60)
plt.show()

#Plot average number per copper atom
'''
coppers=["3su","4su","3pxu","3pyu","3pzu",'3dxyu','3dyzu','3dz2u','3dxzu','3dx2y2u',
"3sd","4sd","3pxd","3pyd","3pzd",'3dxyd','3dyzd','3dz2d','3dxzd','3dx2y2d']
ncu=np.sum(df[coppers],axis=1)
plt.plot(ncu,df['E'],'o')
plt.xlabel("n_cu")
plt.ylabel("E-CHK, eV")
plt.show()
'''
