#USED TO PLOT EXCITATIONS 
#N AND DOUBLE OCCUPATIONS

import numpy as np 
import matplotlib.pyplot as plt
import pickle 
import pandas as pd 
import seaborn as sns
import statsmodels.formula.api as sm
from sklearn.metrics import r2_score
sns.set(style="ticks")

with open("gendata.pickle", 'rb') as handle:
  data = pickle.load(handle)

#Energies
e=np.array(data['energy'])
e-=e[0]
e*=27.2114

#Array of occupations 
ncu=10
no=4
nsr=1

#cu nn
cunns=[
[3,4,1,6],
[0,5,7,2],
[1,6,4,3],
[0,7,5,2],
[0,2,5,7],
[1,6,3,4],
[2,5,0,7],
[1,6,3,4]
]



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

  #Cu nn
  cunn=0
  tcu=np.diag(rdms[0])[:8*ncu]+np.diag(rdms[1])[:8*ncu]
  tcu=np.split(tcu,8)
  for i in range(8):
    for j in cunns[i]:
      cunn+=np.abs(tcu[i][9]-tcu[j][9])
  cunn/=(8.*4.)

  ret=np.array(tmp[0])+np.array(tmp[1])
  ret=ret.tolist()
  ret.append(cunn)
  print(ret[-1])
  n.append(np.array(ret))
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
[2,4,9,10],
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
[1,-1,-1,1],
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

z=0
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
      #if(z==4): print("4:  ",signh[j][p]*(rdms[0][oi,cui]+rdms[0][cui,oi]+rdms[1][oi,cui]+rdms[1][cui,oi]))
      #if(z==16): print("16: ",signh[j][p]*(rdms[0][oi,cui]+rdms[0][cui,oi]+rdms[1][oi,cui]+rdms[1][cui,oi]))
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
        #if(z==4): print("4:  ",sgn[y]*(rdms[0][oi1,oi2]+rdms[0][oi2,oi1]+rdms[1][oi1,oi2]+rdms[1][oi2,oi1]))
        #if(z==16): print("16: ",sgn[y]*(rdms[0][oi1,oi2]+rdms[0][oi2,oi1]+rdms[1][oi1,oi2]+rdms[1][oi2,oi1]))
    else:
      oi1=ncu*8+no*x+1 #px
      for y in range(4): #NN
        oi2=(ncu*8+no*nnhc[x][y]+2) #py
        cctmp+=sgn[y]*(rdms[0][oi1,oi2]+rdms[0][oi2,oi1]+rdms[1][oi1,oi2]+rdms[1][oi2,oi1])
        #if(z==4): print("4:  ",sgn[y]*(rdms[0][oi1,oi2]+rdms[0][oi2,oi1]+rdms[1][oi1,oi2]+rdms[1][oi2,oi1]))
        #if(z==16): print("16: ",sgn[y]*(rdms[0][oi1,oi2]+rdms[0][oi2,oi1]+rdms[1][oi1,oi2]+rdms[1][oi2,oi1]))
  #X
  for x in range(16):
    oi1=0
    oi2=0
    oi1=ncu*8+no*x+1 #px
    for y in range(4): #NN
      oi2=(ncu*8+no*nnhc[x][y]+1) #px
      cxytmp-=(rdms[0][oi1,oi2]+rdms[0][oi2,oi1]+rdms[1][oi1,oi2]+rdms[1][oi2,oi1])
      #if(z==4): print("4:  ",-(rdms[0][oi1,oi2]+rdms[0][oi2,oi1]+rdms[1][oi1,oi2]+rdms[1][oi2,oi1]))
      #if(z==16): print("16: ",-(rdms[0][oi1,oi2]+rdms[0][oi2,oi1]+rdms[1][oi1,oi2]+rdms[1][oi2,oi1]))
    
  #Y
  for x in range(16):
    oi1=0
    oi2=0
    oi1=ncu*8+no*x+2 #px
    for y in range(4): #NN
      oi2=(ncu*8+no*nnhc[x][y]+2) #px
      cxytmp-=(rdms[0][oi1,oi2]+rdms[0][oi2,oi1]+rdms[1][oi1,oi2]+rdms[1][oi2,oi1])
      #if(z==4): print("4:  ",-(rdms[0][oi1,oi2]+rdms[0][oi2,oi1]+rdms[1][oi1,oi2]+rdms[1][oi2,oi1]))
      #if(z==16): print("16: ",-(rdms[0][oi1,oi2]+rdms[0][oi2,oi1]+rdms[1][oi1,oi2]+rdms[1][oi2,oi1]))
  
  cxy.append(cxytmp)
  ch.append(chtmp)
  cc.append(cctmp)
  z+=1

ch=np.array(ch)
cc=np.array(cc)
cxy=np.array(cxy)

#Total data frame
hue=np.zeros(len(e))+4
for i in range(len(hue)):
  if(data['symm'][i]=='xy'): hue[i]=1
  elif(data['symm'][i]=='pi'): hue[i]=2
  elif(data['symm'][i]=='sig'): hue[i]=3
  else: pass

n1=n[:,9]-n[:,14]
n2=n1
Xtot=np.concatenate(((e)[:,np.newaxis],n,n1[:,np.newaxis],n2[:,np.newaxis],u,ch[:,np.newaxis],cc[:,np.newaxis],cxy[:,np.newaxis],hue[:,np.newaxis]),axis=1)
cols=[
"E",
"3s","4s","3px","3py","3pz",'3dxy','3dyz','3dz2','3dxz','3dx2y2','cunn',
'2s','2px','2py','2pz','2psig','2ppi','5s','n1','n2',
"3sU","4sU","3pxU","3pyU","3pzU",'3dxyU','3dyzU','3dz2U','3dxzU','3dx2y2U',
'2sU','2pxU','2pyU','2pzU','5sU',
'ts','tp','txy','symm'
]

#plt.hist(n[:,-1])
#plt.show()

dftot=pd.DataFrame(data=Xtot,columns=cols)
#Energy, all hoppings, all occupations, all U
#sns.pairplot(dftot,vars=["E","2psig","2ppi","3dx2y2","3dx2y2U","ts","tp","txy"],hue='symm')
sns.pairplot(dftot,vars=["E","ts","2psig","2ppi"],hue='symm')
plt.show()
exit(0)

weights=np.ones(len(e))
weights[e<2.0]=(len(e)/7)
regdf=pd.DataFrame({"E":e,"A":ch,"B":cc,"O":cxy,"C":n[:,9],"D":n[:,14],"F":n[:,9]-n[:,14]-n[:,15],"G":u[:,9]})
result=sm.wls(formula="E~A+F",data=regdf[hue==3],weights=weights[hue==3]).fit()
print(result.summary())
pred=result.predict(regdf)
#plt.plot(pred[hue==4],e[hue==4],'mo')
plt.plot(pred[hue==3],e[hue==3],'ro')
'''
plt.plot(pred[hue==2],e[hue==2],'go')
plt.plot(pred[hue==1],e[hue==1],'bo')
'''
plt.plot(e,e,'k-')
plt.show()
exit(0)

Xtot=np.concatenate((e[:,np.newaxis],n,u,ch[:,np.newaxis],cc[:,np.newaxis],cxy[:,np.newaxis],(e-pred)[:,np.newaxis],hue[:,np.newaxis]),axis=1)
cols=[
"E",
"3s","4s","3px","3py","3pz",'3dxy','3dyz','3dz2','3dxz','3dx2y2',
'2s','2px','2py','2pz','2psig','2ppi','5s',
"3sU","4sU","3pxU","3pyU","3pzU",'3dxyU','3dyzU','3dz2U','3dxzU','3dx2y2U',
'2sU','2pxU','2pyU','2pzU','5sU',
'ts','tp','txy','res1','symm'
]
dftot=pd.DataFrame(data=Xtot,columns=cols)
sns.pairplot(dftot,vars=["res1","ts","2ppi"],hue='symm')
plt.show()
