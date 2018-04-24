#Systematic fitting
import numpy as np
import json
import matplotlib.pyplot as plt
from tb_fit import tb 
from sigTB_calc import sigTB
import scipy.optimize
from sklearn.metrics import r2_score

##########################################################################################################
#PBE0 data
d0=[[],[]]

#PBE0 bands
bands=json.load(open("pbe0_bands.p","r"))

j=0
for state in ['col0d','flp2d','blk0d','flp0d','bcol2d']:
  for i in range(1,4):
    d0[0]+=bands[state][i][:38]
  j+=1

#PBE0 energies 
E=np.array([-1.8418601953776E+03,-1.8418566128030E+03 ,-1.8418522579094E+03,-1.8418509420788E+03, -1.8418497382121E+03])
E-=E[0]
E*=27.2
E=E.tolist()
d0[1]=E

##########################################################################################################
#TB data

def dp(t,tt,K,J):
  ret=[[],[]] #Returned vector
  d=10  #fermi expulsion 

  #Calculate TB bands
  states=["COL0","FLP2","BLK0","FLP0","BCOL2"]
  for n in range(len(states)):
    e=[]
    N=12
    x=np.linspace(0,np.pi,N)
    y=np.linspace(0,np.pi,N)
    for i in range(N):
        e.append(tb(t,tt,K,d,x[0],y[i],states[n])[1])
    N=12
    x=np.linspace(0,np.pi,N)
    y=np.linspace(0,np.pi,N)
    for i in range(1,N):
        e.append(tb(t,tt,K,d,x[i],y[N-1],states[n])[1])
    N=16
    x=np.linspace(0,np.pi,N)
    y=np.linspace(0,np.pi,N)
    for i in range(1,N):
        e.append(tb(t,tt,K,d,x[N-1-i],y[N-1-i],states[n])[1])
    e=np.array(e)
    ret[0]+=(e[:,0]).tolist()+(e[:,1]).tolist()+(e[:,2]).tolist() #3 bands
  
  #Calculate energies  
  sigJ=[0,0,0,-2,0]
  tmp=[]
  for n in range(len(sigJ)): #FLP, COL, CHK
    tmp+=[sigTB(t,tt,K,d,states[n])+J*sigJ[n]]
  tmp=np.array(tmp)-tmp[0]
  ret[1]=tmp.tolist()

  return ret

##########################################################################################################
#Cost function
def cost(t,tt,K,J,we,a1,a2,a3,a4,a5,b):
  
  #DEBUG
  print(t,tt,K,J,a1,a2,a3,a4,a5,b)

  dpp=dp(t,tt,K,J)
  dpp[0]=np.array(dpp[0])
  dpp[0][:114]+=a1
  dpp[0][114:228]+=a2
  dpp[0][228:342]+=a3
  dpp[0][342:456]+=a4
  dpp[0][456:]+=a5
  dpp[1]=np.array(dpp[1])+b

  delta=[[],[]]
  delta[0]=dpp[0]-d0[0]
  delta[1]=dpp[1]-d0[1]

  return np.dot(delta[0],delta[0])/len(delta[0])+we*np.dot(delta[1],delta[1])/len(delta[1])

#R2 function
def r2(t,tt,K,J,a1,a2,a3,a4,a5,b):
  dpp=dp(t,tt,K,J)
  dpp[1]=np.array(dpp[1])+b
  dpp[0]=np.array(dpp[0])
  consts=[a1,a2,a3,a4,a5]
  for i in range(5):
    dpp[0][i*38*3:(i+1)*38*3]+=consts[i]
  return (r2_score(d0[0],dpp[0]), r2_score(d0[1],dpp[1]))


##########################################################################################################
#Run
'''
welist=[100]
xlist=[]
r2list=[[],[]]
for we in welist:
  print("########################################################################################")
  print("we="+str(we))

  #cons=({'type':'ineq','fun':lambda x: 0.10-x[3]},{'type':'ineq','fun':lambda x: x[0]},{'type':'ineq','fun':lambda x: x[1]})
  #res=scipy.optimize.minimize(lambda x: cost(x[0],x[1],x[2],x[3],we,x[4],x[5],x[6],x[7],x[8],x[9]),(1.0,0.4,-0.1,0.2,we,0,0,0,0,0,0),constraints=cons)
  res=scipy.optimize.minimize(lambda x: cost(x[0],x[1],x[2],x[3],we,x[4],x[5],x[6],x[7],x[8],x[9]),(1.0,0.4,-0.1,0.2,we,0,0,0,0,0,0))
  xlist.append(res.x) #t,tt,K,J,as,b
  t,tt,K,J,a1,a2,a3,a4,a5,b=res.x
  r2s=r2(t,tt,K,J,a1,a2,a3,a4,a5,b)
 
  r2list[0].append(r2s[0])
  r2list[1].append(r2s[1])
  print("parms: "+str(res.x)) #t,tt,K,J,m,a,b
  print("r2: "+str(r2s[0])+","+str(r2s[1]))

print("########################################################################################")
print("FINAL")
print(xlist)
print(r2list[0])
print(r2list[1])

'''
'''
t,tt,K,J,a1,a2,a3,a4,a5,b=\
[0.94269986323, 0.510298145545, -0.217948885516, 0.1, 1.29099959289, 0.881768856453, 1.28454932116, 0.666130857598, 1.00406073977, -0.0359622382477]
dpp=dp(t,tt,K,J)
print(r2(t,tt,K,J,a1,a2,a3,a4,a5,b))

plt.suptitle("CONST 1: t,tt,K,J="+str([t,tt,K,J])) 

dpp[0]=np.array(dpp[0])
dpp[1]=np.array(dpp[1])
const=[a1,a2,a3,a4,a5]
#5 sets of bands + 1 energy plot
for i in range(5):
  plt.subplot(230+i+1)
  plt.plot(d0[0][i*38*3:(i+1)*38*3],'go')
  plt.plot(dpp[0][i*38*3:(i+1)*38*3]+const[i],'bo')

plt.subplot(236)
plt.title("Energies")
plt.ylabel("E_pred")
plt.xlabel("E_PBE0")
plt.plot(d0[1],d0[1],'g-')
plt.plot(d0[1],dpp[1]+b,'bo')
plt.show()
'''

