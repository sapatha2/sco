import numpy as np 
import matplotlib.pyplot as plt
import pickle 
import pandas as pd 
import seaborn as sns
import statsmodels.formula.api as sm
sns.set(style="ticks")

with open('gendata.pickle', 'rb') as handle:
  data = pickle.load(handle)

#Energies
e=np.array(data['energy'])
e-=e[0]
e*=27.2114

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
    to=np.diag(rdms[s])[8*ncu:8*ncu+16*no]
    to=np.split(to,16)
    o=np.zeros(6)
    o[:4]=np.sum(to,axis=0)
    for i in range(16):
      if(i<8): 
        o[4]+=to[i][1] #sigma
        o[5]+=to[i][2] #pi
      else:
        o[4]+=to[i][2] #sigma
        o[5]+=to[i][1] #pi
    #Sr part
    sr=np.diag(rdms[s])[8*ncu+16*no:8*ncu+16*no+8*nsr]
    sr=np.split(sr,8)
    sr=np.sum(sr,axis=0)
    
    tmp.append(np.concatenate((cu,o,sr),axis=None))
  n.append(np.array(tmp[0])+np.array(tmp[1]))
n=np.array(n)

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
  o=np.split(o,16) #Split amongst the 16 atoms
  o=np.sum(o,axis=0)

  #Sr part
  sr=nund[8*ncu+16*no:8*ncu+16*no+8*nsr]
  sr=np.split(sr,8)
  sr=np.sum(sr,axis=0)
   
  u.append(np.concatenate((cu,o,sr),axis=None))

u=np.array(u)

#Hopping calculations 
ch=[] #Hybridized hopping (Cu d and sigma, no curl)
cc=[] #Curly hopping
cxy=[] #x hopping
nnh=[[0,6,8,11],
[1,7,8,9],
[2,4,9,15],
[3,5,10,11],
[0,4,12,15],
[1,5,12,13],
[2,6,13,14],
[3,7,14,15]] #Nearest neighbors oxygens for a given copper
signh=[[-1,1,1,-1],
[-1,1,-1,1],
[-1,1,-1,1],
[-1,1,-1,1],
[1,-1,1,-1],
[1,-1,-1,1],
[1,-1,1,-1],
[1,-1,-1,1]] #Hopping signs for neighbor oxygens

nnhc=[[15,12,8,11],
[12,13,9,8],
[13,14,10,9],
[14,15,11,10],
[9,10,12,15],
[10,11,13,12],
[11,8,14,13],
[8,9,15,14],
[0,1,7,6],
[1,2,4,7],
[2,3,5,4],
[3,0,6,5],
[4,5,1,0],
[5,6,2,1],
[6,7,3,2],
[7,4,0,3]]
sgn=[1,-1,1,-1]

ncu=10
no=4
nsr=1

for rdms in data['1rdm']:
  chtmp=0
  cctmp=0
  cxytmp=0
  
  #SIGMA
  for j in range(8):
    cui=ncu*j+9
    p=0
    for k in nnh[j]:
      oi=np.nan
      if(k<8): oi=ncu*8+no*k+1 #px
      else: oi=ncu*8+no*k+2    #py
      chtmp+=signh[j][p]*(rdms[0][oi,cui]+rdms[0][cui,oi]+rdms[1][oi,cui]+rdms[1][cui,oi])
      p+=1
  
  #PI
  for x in range(16):
    oi1=0
    oi2=0
    if(x<8): 
      oi1=ncu*8+no*x+2 #py
      for y in range(4): #NN
        oi2=(ncu*8+no*nnhc[x][y]+1) #px
        cctmp+=sgn[y]*(rdms[0][oi1,oi2]+rdms[0][oi2,oi1]+rdms[1][oi1,oi2]+rdms[1][oi2,oi1])
    else:
      oi1=ncu*8+no*x+1 #px
      for y in range(4): #NN
        oi2=(ncu*8+no*nnhc[x][y]+2) #py
        cctmp+=sgn[y]*(rdms[0][oi1,oi2]+rdms[0][oi2,oi1]+rdms[1][oi1,oi2]+rdms[1][oi2,oi1])
    
  #X
  for x in range(16):
    oi1=0
    oi2=0
    oi1=ncu*8+no*x+1 #px
    for y in range(4): #NN
      oi2=(ncu*8+no*nnhc[x][y]+1) #px
      cxytmp+=(rdms[0][oi1,oi2]+rdms[0][oi2,oi1]+rdms[1][oi1,oi2]+rdms[1][oi2,oi1])

  #Y
  for x in range(16):
    oi1=0
    oi2=0
    oi1=ncu*8+no*x+2 #px
    for y in range(4): #NN
      oi2=(ncu*8+no*nnhc[x][y]+2) #px
      cxytmp+=(rdms[0][oi1,oi2]+rdms[0][oi2,oi1]+rdms[1][oi1,oi2]+rdms[1][oi2,oi1])
 
  cxy.append(cxytmp)
  ch.append(chtmp)
  cc.append(cctmp)
ch=np.array(ch)
cc=np.array(cc)
cxy=np.array(cxy)

#Total data frame
Xtot=np.concatenate((e[:,np.newaxis],n,u,ch[:,np.newaxis],cc[:,np.newaxis],cxy[:,np.newaxis]),axis=1)
cols=[
"E",
"3s","4s","3px","3py","3pz",'3dxy','3dyz','3dz2','3dxz','3dx2y2',
'2s','2px','2py','2pz','2psig','2ppi','5s',
"3sU","4sU","3pxU","3pyU","3pzU",'3dxyU','3dyzU','3dz2U','3dxzU','3dx2y2U',
'2sU','2pxU','2pyU','2pzU','5sU',
'ts','tp','txy'
]
dftot=pd.DataFrame(data=Xtot,columns=cols)
sns.pairplot(dftot,vars=["E","ts","tp","txy","2psig","2ppi","3dx2y2U"])
plt.show()

'''
plt.title("Weight LS, ts,tp,nd,ns,np, and Ud")
weights=np.ones(len(e))
weights[e<2.0]=(len(e)/7)**2
regdf=pd.DataFrame({"E":e,"A":ch,"B":cc,"C":n[:,9],"D":n[:,14],"F":n[:,15],"G":u[:,9]})
result=sm.wls(formula="E~A+G+C",data=regdf,weights=weights).fit()
#result=sm.wls(formula="E~A+B+C+D+F+G",data=regdf,weights=weights).fit()
#result=sm.wls(formula="E~A+G",data=regdf,weights=weights).fit()
print(result.summary())
pred=result.predict(regdf)
plt.plot(pred,e,'bo')
plt.plot(e,e,'g-')
plt.show()
'''

'''
Xtot=np.concatenate((e[:,np.newaxis],n,u,ch[:,np.newaxis],cc[:,np.newaxis],(e-pred)[:,np.newaxis]),axis=1)
cols=[
"E",
"3s","4s","3px","3py","3pz",'3dxy','3dyz','3dz2','3dxz','3dx2y2',
'2s','2px','2py','2pz','2psig','2ppi','5s',
"3sU","4sU","3pxU","3pyU","3pzU",'3dxyU','3dyzU','3dz2U','3dxzU','3dx2y2U',
'2sU','2pxU','2pyU','2pzU','5sU',
'td','tc','res1'
]
dftot=pd.DataFrame(data=Xtot,columns=cols)

sns.pairplot(dftot,vars=["res1","td","tc","3dx2y2","2px","2py","2psig","2ppi","3dx2y2U"])
plt.show()
'''
