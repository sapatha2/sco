import numpy as np 
import matplotlib.pyplot as plt
import pickle 
import pandas as pd 
import seaborn as sns
sns.set(style="ticks")

with open('gendata.pickle', 'rb') as handle:
#with open('gendata_noO.pickle','rb') as handle:
  data = pickle.load(handle)

N=len(data['energy'])
#Plot energies
e=np.array(data['energy'])
e-=e[0]
e*=27.2

#SORTEDE.PDF
'''
plt.plot(sorted(e),'o')
plt.ylabel("Energy - CHK, eV")
plt.xlabel("State")
plt.show()
'''

#TRACE.PDF
'''
#Plot nelec of 1rdm
rdms=data['1rdm']
nelec=[np.trace(x[0])+np.trace(x[1]) for x in rdms]
plt.plot(nelec,'o')
plt.ylabel("Tr(up) + Tr(dn)")
plt.xlabel("Excitation")
plt.show()
'''

#Array of occupations 
ncu=10
no=4
nsr=1

n=[]
i=0
for rdms in data['1rdm']:
  tmp=[]
  for s in range(2):
    #Copper part 
    cu=np.diag(rdms[s])[:8*ncu]
    cu=np.split(cu,8)
    cu=np.sum(cu,axis=0)
  
    #Oxygen part
    o=np.diag(rdms[s])[8*ncu:8*ncu+16*no]
    o=np.split(o,16)
    o=np.sum(o,axis=0)
  
    #Sr part
    sr=np.diag(rdms[s])[8*ncu+16*no:8*ncu+16*no+8*nsr]
    sr=np.split(sr,8)
    sr=np.sum(sr,axis=0)
    
    tmp.append(np.concatenate((cu,o,sr),axis=None))
  n.append(np.array(tmp[0])+np.array(tmp[1]))

n=np.array(n)
X=np.concatenate((e[:,np.newaxis],n),axis=1)
cols=[
"E",
"3s","4s","3px","3py","3pz",'3dxy','3dyz','3dz2','3dxz','3dx2y2',
'2s','2px','2py','2pz','5s'
]
df=pd.DataFrame(data=X,columns=cols)

#VARN.PDF
'''
i=0
for col in cols[1:]:
  plt.plot(np.ones(len(df[col]))*i,df[col],'o')
  i+=1
plt.ylabel("n")
plt.xticks(np.arange(len(cols[1:])),cols[1:])
plt.show()
'''

#EVSNPAIRPLOT.PDF
'''
g=sns.pairplot(df,vars=["E","4s","3dx2y2","2s","2px","2py"])
for i, j in zip(*np.triu_indices_from(g.axes, 1)):
    g.axes[i, j].set_visible(False)
plt.show()
'''

#NREG.PDF
'''
import statsmodels.formula.api as sm
regdf=pd.DataFrame({"E":e,"A":n[:,9],"B":n[:,11]+n[:,12]})
result=sm.ols(formula="E ~ A + B",data=regdf).fit()
print(result.params)
print(result.summary())
'''

#Array of double occupations 
ncu=10
no=4
nsr=1

u=[]
i=0
for nund in data['2rdm']:
  #Copper part 
  cu=nund[:8*ncu]
  cu=np.split(cu,8)
  cu=np.sum(cu,axis=0)

  #Oxygen part
  o=nund[8*ncu:8*ncu+16*no]
  o=np.split(o,16)
  o=np.sum(o,axis=0)

  #Sr part
  sr=nund[8*ncu+16*no:8*ncu+16*no+8*nsr]
  sr=np.split(sr,8)
  sr=np.sum(sr,axis=0)
   
  u.append(np.concatenate((cu,o,sr),axis=None))

u=np.array(u)
XU=np.concatenate((e[:,np.newaxis],u),axis=1)

cols=[
"E",
"3sU","4sU","3pxU","3pyU","3pzU",'3dxyU','3dyzU','3dz2U','3dxzU','3dx2y2U',
'2sU','2pxU','2pyU','2pzU','5sU'
]
dfU=pd.DataFrame(data=XU,columns=cols)

#VARU.PDF
'''
i=0
for col in cols[1:]:
  plt.plot(np.ones(len(dfU[col]))*i,dfU[col],'o')
  i+=1
plt.ylabel("nu*nd")
plt.xticks(np.arange(len(cols[1:])),cols[1:])
plt.show()
'''

#EVSUPAIRPLOT.PDF
g=sns.pairplot(dfU,vars=["E","4sU","3dx2y2U","2sU","2pxU","2pyU"])
plt.show()
