import numpy as np 
from numpy import exp 
import matplotlib.pyplot as plt
import json
import scipy.optimize
from sklearn.metrics import r2_score

################################################################################3
#BAND CALCULATION 

def NN(t,x,y):
  M=-t*np.array([[0,0,1,0,exp(1j*y),0,exp(-1j*x),exp(1j*(y-x))],
                 [0,0,1,0,1,0,exp(-1j*x),exp(-1j*x)],
                 [0,0,0,1,0,1,0,0],
                 [0,0,0,0,exp(1j*y),0,1,exp(1j*y)],
                 [0,0,0,0,0,1,0,0],
                 [0,0,0,0,0,0,1,1],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0]])
  M+=M.conj().T
  return M

def NNN(tt,x,y):
  M=tt*np.array([[0,1+exp(1j*y),0,1+exp(-1j*x),0,0,0,0],
                 [0,0,0,0,0,1+exp(-1j*x),0,0],
                 [0,0,0,0,1+exp(1j*y),0,1+exp(-1j*x),0],
                 [0,0,0,0,0,1+exp(1j*y),0,0],
                 [0,0,0,0,0,0,0,1+exp(-1j*x)],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,1+exp(1j*y)],
                 [0,0,0,0,0,0,0,0]])
  M+=M.conj().T
  return M

def col0(K,d):
  up=np.array([[d,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,d,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,d,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,d]])
  dn=np.array([[0,0,0,0,0,0,0,0],
               [0,d,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,d,0,0,0,0],
               [0,0,0,0,d,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,d,0],
               [0,0,0,0,0,0,0,0]])
  return [up,dn]

def flp2(K,d):
  up=np.zeros((8,8))
  dn=np.array([[-K,0,0,0,0,0,0,0],
               [0,d-K,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,d-K,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,-K,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0]])
  return [up,dn]

def blk0(K,d):
  up=np.array([[d,0,0,0,0,0,0,0],
               [0,d,0,0,0,0,0,0],
               [0,0,d,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,d,0],
               [0,0,0,0,0,0,0,0]])
  dn=np.array([[0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,d,0,0,0,0],
               [0,0,0,0,d,0,0,0],
               [0,0,0,0,0,d,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,d]])
  return [up,dn]
def flp0(K,d):
  up=np.zeros((8,8))
  dn=np.array([[d-K,0,0,0,0,0,0,0],
               [0,d-K,0,0,0,0,0,0],
               [0,0,0.5*K,0,0,0,0,0],
               [0,0,0,d-K,0,0,0,0],
               [0,0,0,0,0.5*K,0,0,0],
               [0,0,0,0,0,-K,0,0],
               [0,0,0,0,0,0,0.5*K,0],
               [0,0,0,0,0,0,0,0.5*K]])
  return [up,dn]

def bcol2(K,d):
  up=np.zeros((8,8))
  dn=np.array([[0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,d-K,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,-K,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,d-K,0],
               [0,0,0,0,0,0,0,-K]])
  return [up,dn]

def tb(t,tt,K,x,y,state):
  d=10
  H=[np.zeros((8,8))+0j,np.zeros((8,8))+0j]
  if(state=="col0_n"):  
    H[0]+=col0(K,d)[0]
    H[1]+=col0(K,d)[1]
  elif(state=="flp2_n"):
    H[0]+=flp2(K,d)[0]
    H[1]+=flp2(K,d)[1]
  elif(state=="achn2_n"):
    H[0]+=flp2(K,d)[0]
    H[1]+=flp2(K,d)[1]
  elif(state=="blk0_n"):
    H[0]+=blk0(K,d)[0]
    H[1]+=blk0(K,d)[1]
  elif(state=="flp0_n"):
    H[0]+=flp0(K,d)[0]
    H[1]+=flp0(K,d)[1]
  elif(state=="bcol2_n"):
    H[0]+=bcol2(K,d)[0]
    H[1]+=bcol2(K,d)[1]
  H[0]+=NN(t,x,y)+NNN(tt,x,y)
  H[1]+=NN(t,x,y)+NNN(tt,x,y)
  wu,vr=np.linalg.eigh(H[0])
  wd,vr=np.linalg.eigh(H[1])
  return [wu,wd]

################################################################################3
#BAND OPTIMIZATION 
def dp(t,tt,K,a,state):
  d=10
  e=[]
  eu=[]
  N=12
  x=np.linspace(0,np.pi,N)
  y=np.linspace(0,np.pi,N)
  for i in range(N-1):
    e.append(tb(t,tt,K,x[0],y[i],state)[1])
  N=12
  x=np.linspace(0,np.pi,N)
  y=np.linspace(0,np.pi,N)
  for i in range(N-1):
    e.append(tb(t,tt,K,x[i],y[N-1],state)[1])
  N=16
  x=np.linspace(0,np.pi,N)
  y=np.linspace(0,np.pi,N)
  for i in range(N):
    e.append(tb(t,tt,K,x[N-1-i],y[N-1-i],state)[1])

  #Plot PBE0 bands and TB bands
  e=np.array(e)+a
  ret=[]
  if(state in ["col0_n","blk0_n","chk0_n"]):
    ret+=e[:,0].tolist()+e[:,1].tolist()+e[:,0].tolist()+e[:,1].tolist()
  else:
    for i in range(4):
      ret+=e[:,i].tolist()
  return np.array(ret)

def cost(t,tt,K,a,state,lim):
  bands=json.load(open("pbe0_bands.p","r"))
  d0=[]
  for i in range(4):
    d0+=bands[state.lower()][i][:38]
  d0=np.array(d0)
  dpp=dp(t,tt,K,a,state)
  delta=dpp-d0

  logi=np.logical_or(np.logical_and(d0>=-lim,d0<=lim),np.logical_and(dpp>=-lim,dpp<=lim))
  ind=np.where(logi)
  delta=delta[ind]
  return np.dot(delta,delta)

def r2(t,tt,K,a,state):
  bands=json.load(open("pbe0_bands.p","r"))
  d0=[]
  for i in range(4):
    d0+=bands[state.lower()][i][:38]
  d0=np.array(d0)
  dpp=dp(t,tt,K,a,state)
  return r2_score(d0,dpp) 

def plot(t,tt,K,a,state):
  bands=json.load(open("pbe0_bands.p","r"))
  d0=[]
  for i in range(4):
    d0+=bands[state.lower()][i][:38]
  d0=np.array(d0)
  dpp=dp(t,tt,K,a,state)
  
  plt.ylabel("E-Ef, eV")
  plt.plot(d0,'go')
  plt.plot(dpp,'b.')

def cost3(t,tt,K,a1,a2,a3,lim):
  ret=0
  ret+=cost(t,tt,K,a1,"col0_n",lim)
  ret+=cost(t,tt,K,a2,"achn2_n",lim)
  ret+=cost(t,tt,K,a3,"flp0_n",lim)
  return ret

def r23(t,tt,K,a1,a2,a3):
  d0=[]
  dpp=[]
  for state in ["col0_n","achn2_n","flp0_n"]:
    bands=json.load(open("pbe0_bands.p","r"))
    for i in range(4):
      d0+=bands[state.lower()][i][:38]
  
  dpp+=dp(t,tt,K,a1,"col0_n").tolist()
  dpp+=dp(t,tt,K,a2,"achn2_n").tolist()
  dpp+=dp(t,tt,K,a3,"flp0_n").tolist()
  return r2_score(d0,dpp) 

def cost4(t,tt,K,a1,a2,a3,a4,lim):
  #print(t,tt,K)
  ret=0
  ret+=cost(t,tt,K,a1,"col0_n",lim)
  ret+=cost(t,tt,K,a2,"achn2_n",lim)
  ret+=cost(t,tt,K,a3,"flp0_n",lim)
  ret+=cost(t,tt,K,a4,"bcol2_n",lim)
  return ret

def r24(t,tt,K,a1,a2,a3,a4):
  d0=[]
  dpp=[]
  bands=json.load(open("pbe0_bands.p","r"))
  for state in ["col0_n","achn2_n","flp0_n","bcol2_n"]:
    for i in range(4):
      d0+=bands[state.lower()][i][:38]

  dpp+=dp(t,tt,K,a1,"col0_n").tolist()
  dpp+=dp(t,tt,K,a2,"achn2_n").tolist()
  dpp+=dp(t,tt,K,a3,"flp0_n").tolist()
  dpp+=dp(t,tt,K,a4,"bcol2_n").tolist()
  return r2_score(d0,dpp) 

def plot4(t,tt,K,a1,a2,a3,a4):
  j=0
  const=[a1,a2,a3,a4]
  for state in ["col0_n","achn2_n","flp0_n","bcol2_n"]:
    plt.subplot(230+j+1)
    plt.title(state+" bands")
    plot(t,tt,K,const[j],state)
    j+=1
  
  #plt.show()

def cost5(t,tt,K,a1,a2,a3,a4,a5,lim):
  #print(t,tt,K)
  ret=0
  ret+=cost(t,tt,K,a1,"COL0",lim)
  ret+=cost(t,tt,K,a2,"FLP2",lim)
  ret+=cost(t,tt,K,a3,"FLP0",lim)
  ret+=cost(t,tt,K,a4,"BCOL2",lim)
  ret+=cost(t,tt,K,a5,"CHK0",lim)
  return ret

def r25(t,tt,K,a1,a2,a3,a4,a5):
  d0=[]
  dpp=[]
  bands=json.load(open("pbe0_bands.p","r"))
  for state in ["COL0","FLP2","FLP0","BCOL2","CHK0"]:
    if(state=="COL0" or state=="CHK0"):
      for i in range(2):
        d0+=bands[state.lower()+"d"][i][:38]
        d0+=bands[state.lower()+"d"][i][:38]
    else:
      for i in range(4):
        d0+=bands[state.lower()+"d"][i][:38]

  dpp+=dp(t,tt,K,a1,"COL0").tolist()
  dpp+=dp(t,tt,K,a2,"FLP2").tolist()
  dpp+=dp(t,tt,K,a3,"FLP0").tolist()
  dpp+=dp(t,tt,K,a4,"BCOL2").tolist()
  dpp+=dp(t,tt,K,a5,"CHK0").tolist()
  return r2_score(d0,dpp) 

def plot5(t,tt,K,a1,a2,a3,a4,a5):
  j=0
  const=[a1,a2,a3,a4,a5]
  for state in ["COL0","FLP2","FLP0","BCOL2","CHK0"]:
    plt.subplot(230+j+1)
    plot(t,tt,K,const[j],state)
    j+=1

  #plt.show()

################################################################################
#ENERGY OPTIMIZATION
def sigTB(t,tt,K,state):
  d=10
  N=50
  e=[]
  rho=[]
  x=np.linspace(-np.pi+2*np.pi/(N-1),np.pi,N-1)
  y=np.linspace(-np.pi+2*np.pi/(N-1),np.pi,N-1)

  for i in range(N-1):
    for j in range(N-1):
      w=tb(t,tt,K,x[i],y[j],state)
      if(state in ["col0_n","blk0_n","chk0_n"]):
        for k in range(len(w[0])):
          e.append(w[0][k])
          e.append(w[1][k])
      else:
        for k in range(len(w[1])):
          e.append(w[1][k])

  e=sorted(e)
  #plt.plot(e,'b.')
  e=e[:2*(N-1)**2]
  #plt.plot(e,'g.')
  return np.sum(e)/((N-1)**2)

def cost3j(t,tt,K,J,b):
  print(t,tt,K,J)
  sigJ=[0,0,-2]
  E=[]
  ind=0
  for state in ["col0_n","achn2_n","flp0_n"]:
    E.append(sigTB(t,tt,K,state)+J*sigJ[ind])
    ind+=1
  E=np.array(E)
  E+=b
  E-=E[0]
  
  E0=np.array([0,0.08540447488,0.2076015466])
  E0-=E0[0]

  return np.dot(E0-E,E0-E)

def r23j(t,tt,K,J,b):
  sigJ=[0,0,-2]
  E=[]
  ind=0
  for state in ["col0_n","achn2_n","flp0_n"]:
    E.append(sigTB(t,tt,K,state)+J*sigJ[ind])
    ind+=1
  E=np.array(E)
  E+=b
  E-=E[0]
  
  E0=np.array([0,0.08540447488,0.2076015466])
  E0-=E0[0]
  
  return r2_score(E0,E)

def plot3j(t,tt,K,J,b):
  sigJ=[0,0,-2]
  E=[]
  ind=0
  for state in ["col0_n","achn2_n","flp0_n"]:
    E.append(sigTB(t,tt,K,state)+J*sigJ[ind])
    ind+=1
  E=np.array(E)
  E+=b
  E-=E[0]
  
  E0=np.array([0,0.08540447488,0.2076015466])
  E0-=E0[0]
  plt.plot(E0,E,'bo')
  plt.plot(E0,E0,'g')
  #plt.show()

def cost4j(t,tt,K,J,b):
  #print(t,tt,K,J)
  
  sigJ=[0,0,-2,0]
  E=[]
  ind=0
  for state in ["col0_n","achn2_n","flp0_n","bcol2_n"]:
    E.append(sigTB(t,tt,K,state)+J*sigJ[ind])
    ind+=1
  E=np.array(E)
  E+=b
  E-=E[0]

  E0=np.array([0,0.08540447488,0.2076015466,0.2229464728])
  E0-=E0[0]

  return np.dot(E0-E,E0-E)

def r24j(t,tt,K,J,b):
  sigJ=[0,0,-2,0]
  E=[]
  ind=0
  for state in ["col0_n","achn2_n","flp0_n","bcol2_n"]:
    E.append(sigTB(t,tt,K,state)+J*sigJ[ind])
    ind+=1
  E=np.array(E)
  E+=b
  E-=E[0]
  
  E0=np.array([0,0.08540447488,0.2076015466,0.2229464728])
  E0-=E0[0]
  
  return r2_score(E0,E)

def plot4j(t,tt,K,J,b):
  sigJ=[0,0,-2,0]
  E=[]
  ind=0
  for state in ["col0_n","achn2_n","flp0_n","bcol2_n"]:
    E.append(sigTB(t,tt,K,state)+J*sigJ[ind])
    ind+=1
  E=np.array(E)
  E+=b
  E-=E[0]
  
  E0=np.array([0,0.08540447488,0.2076015466,0.2229464728])
  E0-=E0[0]
  plt.plot(E0,E,'bo')
  plt.plot(E0,E0,'g')
  plt.ylabel("E_pred, eV")
  plt.xlabel("E_PBE0, eV")
  plt.title("Total Energy fit")

def cost5j(t,tt,K,J,b):
  #print(t,tt,K,J)
  sigJ=[0,0,-2,0,0]
  E=[]
  ind=0
  for state in ["COL0","FLP2","FLP0","BCOL2","CHK0"]:
    E.append(sigTB(t,tt,K,state)+J*sigJ[ind])
    ind+=1
  E=np.array(E)
  E+=b
  E-=E[0]

  E0=np.array([0,0.106,0.252,0.285,0.412])
  E0-=E0[0]

  return np.dot(E0-E,E0-E)

def r25j(t,tt,K,J,b):
  sigJ=[0,0,-2,0,0]
  E=[]
  ind=0
  for state in ["COL0","FLP2","FLP0","BCOL2","CHK0"]:
    E.append(sigTB(t,tt,K,state)+J*sigJ[ind])
    ind+=1
  E=np.array(E)
  E+=b
  E-=E[0]
  
  E0=np.array([0,0.106,0.252,0.285,0.412])
  E0-=E0[0]
  
  return r2_score(E0,E)

def plot5j(t,tt,K,J,b):
  sigJ=[0,0,-2,0,0]
  E=[]
  ind=0
  for state in ["COL0","FLP2","FLP0","BCOL2","CHK0"]:
    E.append(sigTB(t,tt,K,state)+J*sigJ[ind])
    ind+=1
  E=np.array(E)
  E+=b
  E-=E[0]
  
  E0=np.array([0,0.106,0.252,0.285,0.412])
  E0-=E0[0]
  plt.plot(E0,E,'ro')
  plt.plot(E0,E0,'g')
  #plt.show()

################################################################################3
#BOTH ENERGY AND BANDS: 3 STATES
def cost3_both(t,tt,K,J,a1,a2,a3,b,we,lim):
  print(t,tt,K,J)
  return cost3(t,tt,K,a1,a2,a3,lim)+we*cost3j(t,tt,K,J,b)

#BOTH ENERGY AND BANDS: 4 STATES
def cost4_both(t,tt,K,J,a1,a2,a3,a4,b,we,lim):
  print(t,tt,K,J)
  return cost4(t,tt,K,a1,a2,a3,a4,lim)+we*cost4j(t,tt,K,J,b)

#BOTH ENERGY AND BANDS: 5 STATES
def cost5_both(t,tt,K,J,a1,a2,a3,a4,a5,b,we,lim):
  print(t,tt,K,J)
  return cost5(t,tt,K,a1,a2,a3,a4,a5,lim)+we*cost5j(t,tt,K,J,b)

################################################################################3
#SINGLE STATE BAND RUN
'''
lim=0.5
state="bcol2_n"
res=scipy.optimize.minimize(lambda p: cost(p[0],p[1],p[2],p[3],state,lim),(1.1,0.4,0.0,1.0))
t,tt,K,a=res.x
print(res.x)
print(r2(t,tt,K,a,state))
plot(t,tt,K,a,state)
plt.show()
'''

################################################################################3
#THREE STATE BAND RUN
'''
lim=0.25
res=scipy.optimize.minimize(lambda p: cost3(p[0],p[1],p[2],p[3],p[4],p[5],lim),(1.1,0.4,0.0,1.0,1.0,1.0))
t,tt,K,a1,a2,a3=res.x
print(res.x)
print(r23(t,tt,K,a1,a2,a3))
plot(t,tt,K,a1,"col0_n")
plt.show()
plot(t,tt,K,a2,"achn2_n")
plt.show()
plot(t,tt,K,a3,"flp0_n")
plt.show()
'''

#BAND ONLY CALCULATIONS
#lim=10, R2=0.97
#[ 1.05701702  0.44753253 -0.09978872]

#lim=0.5, R2=0.95
#[0.92336509 0.42020023 0.07449475]

#lim=0.25, R2=0.94
#[0.88701912 0.46070515 0.13790199]

################################################################################3
#FOUR STATE BAND RUN
'''
lim=10
res=scipy.optimize.minimize(lambda p: cost4(p[0],p[1],p[2],p[3],p[4],p[5],p[6],lim),(1.1,0.4,0.0,1.0,1.0,1.0,1.0))
t,tt,K,a1,a2,a3,a4=res.x
print(res.x)
print(r24(t,tt,K,a1,a2,a3,a4))
plot4(t,tt,K,a1,a2,a3,a4)
plt.show()
'''

#lim=10
#[ 1.08978554  0.43633387 -0.26546637  1.45904235  1.03865031  0.7620316
#  1.0468017 ]
#0.9712088073465714

#lim=0.5, GOOD
#[ 0.99297232  0.4093222  -0.37465912  1.37275236  0.99458692  0.72542483
#  0.9585292 ]
#0.9526495950755716

#lim=0.25, GOOD
#[0.95306339  0.43679811 -0.28430107  1.33377059  1.01022471  0.68019914
# 0.98633076]
#0.93527466230021

#lim=0.10
#[1.10003627e+00 4.00000000e-01 3.62710458e-05 1.00000000e+00
# 1.00000000e+00 1.00000000e+00 1.00000000e+00]
#0.8107323700474109

################################################################################3
#THREE STATE ENERGY RUN
'''
constraints=({'type':'eq','fun':lambda p: p[3]-0.18503043029482322})
res=scipy.optimize.minimize(lambda p: cost3j(p[0],p[1],p[2],p[3],p[4]),(1.1,0.4,0.0,0.18503043029482322,0.0),constraints=constraints)
t,tt,K,J,b=res.x
print(res.x)
print(r23j(t,tt,K,J,b))
'''

#BEST ENERGY, NO PRIOR; R2 = 1.0
#t,tt,K,J,b=[1.03863415,  0.47292914, -0.02339709 , 0.27248094 , 0.]
#print(r23j(t,tt,K,J,b))
#plot3j(t,tt,K,J,b)
#plt.show()

#BEST ENERGY, J=0.18503043029482322 PRIOR; R2 = 0.9999
#t,tt,K,J,b=[ 0.9894164,   0.51746822, -0.08044577,  0.18503043,  0.        ]
#print(r23j(t,tt,K,J,b))
#plot3j(t,tt,K,J,b)
#plt.show()

################################################################################3
#FOUR STATE ENERGY RUN
'''
constraints=({'type':'eq','fun':lambda p: p[3]-0.18503043029482322})
res=scipy.optimize.minimize(lambda p: cost4j(p[0],p[1],p[2],p[3],p[4]),(2.0,1.0,0.01,0.18503043029482322,0.0),constraints=constraints)
#res=scipy.optimize.minimize(lambda p: cost4j(p[0],p[1],p[2],p[3],p[4]),(2.0,1.0,0.01,0.40,0.0))
t,tt,K,J,b=res.x
print(res.x)
print(r24j(t,tt,K,J,b))
plot4j(t,tt,K,J,b)
plt.show()
'''

#BEST ENERGY, NO PRIOR; R2=1.0
#t,tt,K,J=[ 1.98188489  1.01511623 -0.04262632  0.45326306  0.        ]

#BEST ENERGY, PRIOR; R2=0.986067917347741
#[ 1.79540117  1.14680344 -0.31993837  0.18503043  0.        ]

################################################################################3
#THREE STATE ENERGY AND BAND RUN 
'''
we=5
lim=0.50
J=0.18503043029482322
constraints=({'type':'eq','fun':lambda p: p[3]-J})
res=scipy.optimize.minimize(lambda p: cost3_both(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],we,lim),(1.1,0.4,0.0,J,1.0,1.0,1.0,0.0),constraints=constraints)
t,tt,K,J,a1,a2,a3,b=res.x
print(res.x)
'''

'''
t,tt,K,J,a1,a2,a3,b=[ 0.87988078,  0.43157432, -0.05390186, 0.18503043,  1.22132379,  0.99872257, 0.54454375,  0.]
print(r23(t,tt,K,a1,a2,a3),r23j(t,tt,K,J,b))
plt.subplot(221)
plot3j(t,tt,K,J,b)
plt.subplot(222)
plot(t,tt,K,a1,"col0_n")
plt.subplot(223)
plot(t,tt,K,a2,"achn2_n")
plt.subplot(224)
plot(t,tt,K,a3,"flp0_n")
plt.show()
'''

#BOTH, we=1, lim=0.25, prior 
#[0.88622686 0.43864643 0.05465994 0.18503043 1.22806282 1.03683781
# 0.49584947 0.        ]
#0.9450445374868651 0.526145193426165

#BOTH, we=5, lim=0.25, prior 
#[ 0.88908953  0.43813217 -0.0363622   0.18503043  1.23195472  1.02190757
#  0.54500791  0.        ]
#0.9392491916901259 0.983381827576002

#BOTH, we=5, lim=0.50, prior 
#[0.90875172 0.4131968  0.02119196 0.18503043 1.24246468 1.02231345
# 0.50763467 0.        ]
#0.9546273936604586 0.4877157608242211

#BOTH, we=10, lim=0.50, prior 
#[ 0.90097909  0.41031054 -0.00223558  0.18503043  1.22937314  1.00567278
#  0.5131108   0.        ]
#0.9523865217300596 0.6394000391527384

#BOTH, we=10, lim=0.25, prior 
#[ 0.87956421  0.42883839 -0.03841918  0.18503043  1.22123757  1.00250862
#  0.53227712  0.        ]
#0.9385360876008545 0.9885378408802482

#BOTH, we=20, lim=0.50, prior 
#[ 0.88921062  0.40860529 -0.00878038  0.18503043  1.21025214  0.99308946
#  0.51162405  0.        ]
#0.9491825205181774 0.7613868819558821

#BOTH, we=20, lim=0.25, prior 
#[ 0.87769923  0.42852605 -0.04546482  0.18503043  1.21907485  1.00558652
#  0.53513253  0.        ]
#0.9364298378821341 0.9961116591046266

#BOTH, we=50, lim=0.50, prior 
#[ 0.87391923  0.40783117 -0.02677096  0.18503043  1.18010164  0.9752992
#  0.51454945  0.        ]
#0.9437164944817021 0.9103911025631894

#BOTH, we=200, lim=0.50, prior 
#[ 0.8477594   0.39872371 -0.04029983  0.18503043  1.12920661  0.94583792
#  0.51041711  0.        ]
#0.9345239643952593 0.9865818818917892

#BOTH, we=500, lim=0.50, prior 
#[ 0.86156764  0.41332302 -0.04896716  0.18503043  1.15348468  0.95567881
#  0.51861259  0.        ]
#0.937887440485707 0.9968717028410259

#BOTH, we=500, lim=0.25, prior 
#[ 0.87988078  0.43157432 -0.05390186  0.18503043  1.22132379  0.99872257
#  0.54454375  0.        ]
#0.9371337025509947 0.9997876553183775

################################################################################3
#FOUR STATE ENERGY AND BAND RUN 
'''
we=10000
lim=0.50
J=0.18503043029482322
constraints=({'type':'eq','fun':lambda p: p[3]-J})
res=scipy.optimize.minimize(lambda p: cost4_both(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],we,lim),(2.0,1.0,-0.5,J,1.0,1.0,1.0,1.0,0.0),constraints=constraints)
t,tt,K,J,a1,a2,a3,a4,b=res.x
print(res.x)
'''
t,tt,K,J,a1,a2,a3,a4,b= [ 0.93874653,  0.47264805, -0.10206233 , 0.18503043  ,1.28444839,  1.06119634, 0.59476486 , 1.08946646  ,0.        ]
print(r24(t,tt,K,a1,a2,a3,a4),r24j(t,tt,K,J,b))
plot4(t,tt,K,a1,a2,a3,a4)
plt.subplot(236)
plot4j(t,tt,K,J,b)
plt.show()

#####################################################
#WITH COL0, NOT BLK0

#FINISH RUNNING AND THEN MAKE THAT TABLE; want 0.5 for each R2e and R2E

#BOTH we=100, lim=0.25, prior 
#[ 1.58343066  0.9948051  -0.25887215  0.18503043  2.18167694  1.91102479
#  1.31666284  1.93669192  0.        ]
#0.6242514879556291 0.9648698000215449

#BOTH we=100, lim=0.50, prior 
#[ 0.8882634   0.40567044 -0.09808818  0.18503043  1.20336692  0.96685219
#  0.55085088  0.99083287  0.        ]
#0.9315087144100165 0.7895741798364578

#BOTH we=200, lim=0.25, prior 
#[ 0.87828059  0.42200776 -0.09710887  0.18503043  1.22280815  0.98073016
#  0.55054733  1.04240884  0.        ]
#0.9148156144989904 0.9145575926551728

#BOTH we=200, lim=0.50, prior 
#[ 0.89751269  0.42287145 -0.10356723  0.18503043  1.22410815  0.98912445
#  0.5393895   0.99178879  0.        ]
#0.9304601929963011 0.8576153690175323

#BOTH we=500, lim=0.25, prior 
#[ 0.91969447  0.45694792 -0.10014718  0.18503043  1.2748363   1.04582093
#  0.59376545  1.09944008  0.        ]
#0.9192312735710522 0.9339367568703769

#BOTH we=500, lim=0.50, prior 
#[ 0.89793111  0.43339174 -0.09960211  0.18503043  1.19311371  0.9777497
#  0.52449217  1.01943263  0.        ]
#0.9273547417568664 0.9095212652493521

#BOTH we=1000, lim=0.25, prior 
#[ 1.64785413  1.03211572 -0.27006195  0.18503043  2.29189917  1.99731728
#  1.38712076  2.01846717  0.        ]
#0.5718093266322289 0.9909235404658929

#BOTH we=1000, lim=0.50, prior 
#[ 0.93874653  0.47264805 -0.10206233  0.18503043  1.28444839  1.06119634
#  0.59476486  1.08946646  0.        ]
#0.9271996617186321 0.9416184370025703

#BOTH we=5000, lim=0.25, prior 
#[ 1.50964107  0.92377376 -0.22739743  0.18503043  2.10254188  1.84598048
#  1.29287689  1.85904995  0.        ]
#0.6931897661350835 0.9961298316427654

#BOTH we=10000, lim=0.25, prior 
#[ 1.36200442  0.80798058 -0.18738722  0.18503043  1.88729016  1.6420394
#  1.05994886  1.6746161   0.        ]
#0.7988394347025202 0.99993453956618

#BOTH we=10000, lim=0.50, prior 
#[ 1.17733699  0.6630243  -0.14222643  0.18503043  1.62269036  1.38558903
#  0.78508355  1.40388237  0.        ]
#0.8925939244657366 0.9937252950342316

#####################################################
#PARETO PLOTS
'''
pareto3=[[0.9450445374868651, 0.526145193426165],
[0.9392491916901259, 0.983381827576002],
[0.9546273936604586, 0.4877157608242211], 
[0.9523865217300596, 0.6394000391527384], 
[0.9385360876008545, 0.9885378408802482],
[0.9491825205181774, 0.7613868819558821], 
[0.9364298378821341, 0.9961116591046266],
[0.9437164944817021, 0.9103911025631894],
[0.9345239643952593, 0.9865818818917892],
[0.937887440485707, 0.9968717028410259],
[0.9371337025509947, 0.9997876553183775]]
pareto3=np.array(pareto3)
'''
pareto4=[[0.9315087144100165, 0.7895741798364578],
[0.9148156144989904, 0.9145575926551728],
[0.9304601929963011, 0.8576153690175323],
[0.9192312735710522, 0.9339367568703769],
[0.9273547417568664, 0.9095212652493521],
[0.5718093266322289, 0.9909235404658929],
[0.9271996617186321, 0.9416184370025703],
[0.6931897661350835, 0.9961298316427654],
[0.7988394347025202, 0.99993453956618],
[0.8925939244657366, 0.9937252950342316]]
pareto4=np.array(pareto4)

plt.plot(pareto4[:,0],pareto4[:,1],'go')
plt.plot(0.9271996617186321, 0.9416184370025703, 'bs')
plt.title("4-state Pareto")
plt.xlabel("R^2 band")
plt.ylabel("R^2 energy")
plt.show()
