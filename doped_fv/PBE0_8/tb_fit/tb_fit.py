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
  if(state=="COL0"):  
    H[0]+=col0(K,d)[0]
    H[1]+=col0(K,d)[1]
  elif(state=="FLP2"):
    H[0]+=flp2(K,d)[0]
    H[1]+=flp2(K,d)[1]
  elif(state=="BLK0"):
    H[0]+=blk0(K,d)[0]
    H[1]+=blk0(K,d)[1]
  elif(state=="FLP0"):
    H[0]+=flp0(K,d)[0]
    H[1]+=flp0(K,d)[1]
  elif(state=="BCOL2"):
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
  for i in range(4):
    ret+=e[:,i].tolist()
  return np.array(ret)

def cost(t,tt,K,a,state,lim):
  #print(t,tt,K)
  bands=json.load(open("pbe0_bands.p","r"))
  d0=[]
  for i in range(4):
    d0+=bands[state.lower()+"d"][i][:38]
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
    d0+=bands[state.lower()+"d"][i][:38]
  d0=np.array(d0)
  dpp=dp(t,tt,K,a,state)
  return r2_score(d0,dpp) 

def plot(t,tt,K,a,state):
  bands=json.load(open("pbe0_bands.p","r"))
  d0=[]
  for i in range(4):
    d0+=bands[state.lower()+"d"][i][:38]
  d0=np.array(d0)
  dpp=dp(t,tt,K,a,state)

  plt.plot(d0,'go-')
  plt.plot(dpp,'bo-')
  #plt.show()

def cost3(t,tt,K,a1,a2,a3,lim):
  ret=0
  ret+=cost(t,tt,K,a1,"FLP2",lim)
  ret+=cost(t,tt,K,a2,"FLP0",lim)
  ret+=cost(t,tt,K,a3,"BCOL2",lim)
  return ret

def r23(t,tt,K,a1,a2,a3):
  d0=[]
  dpp=[]
  for state in ["FLP2","FLP0","BCOL2"]:
    bands=json.load(open("pbe0_bands.p","r"))
    for i in range(4):
      d0+=bands[state.lower()+"d"][i][:38]
  
  dpp+=dp(t,tt,K,a1,"FLP2").tolist()
  dpp+=dp(t,tt,K,a2,"FLP0").tolist()
  dpp+=dp(t,tt,K,a2,"BCOL2").tolist()
  return r2_score(d0,dpp) 

def cost5(t,tt,K,a1,a2,a3,lim):
  pass

def r25(t,tt,K,a1,a2,a3):
  pass

################################################################################
#ENERGY OPTIMIZATION
def sigTB(t,tt,K,state):
  d=10
  N=20
  e=[]
  rho=[]
  x=np.linspace(-np.pi+2*np.pi/(N-1),np.pi,N-1)
  y=np.linspace(-np.pi+2*np.pi/(N-1),np.pi,N-1)

  for i in range(N-1):
    for j in range(N-1):
      w=tb(t,tt,K,x[i],y[j],state)
      for k in range(len(w[1])):
        e.append(w[1][k])

  e=sorted(e)
  e=e[:2*(N-1)**2]
  
  return np.sum(e)/((N-1)**2)

def cost3j(t,tt,K,J,b):
  #print(t,tt,K,J)
  sigJ=[0,-2,0]
  E=[]
  ind=0
  for state in ["FLP2","FLP0","BCOL2"]:
    E.append(sigTB(t,tt,K,state)+J*sigJ[ind])
    ind+=1
  E=np.array(E)
  E+=b
  E-=E[0]
  
  E0=np.array([0.106,0.252,0.285])
  E0-=E0[0]

  return np.dot(E0-E,E0-E)

def r23j(t,tt,K,J,b):
  sigJ=[0,-2,0]
  E=[]
  ind=0
  for state in ["FLP2","FLP0","BCOL2"]:
    E.append(sigTB(t,tt,K,state)+J*sigJ[ind])
    ind+=1
  E=np.array(E)
  E+=b
  E-=E[0]
  
  E0=np.array([0.106,0.252,0.285])
  E0-=E0[0]
  
  return r2_score(E0,E)

def plot3j(t,tt,K,J,b):
  sigJ=[0,-2,0]
  E=[]
  ind=0
  for state in ["FLP2","FLP0","BCOL2"]:
    E.append(sigTB(t,tt,K,state)+J*sigJ[ind])
    ind+=1
  E=np.array(E)
  E+=b
  E-=E[0]
  
  E0=np.array([0.106,0.252,0.285])
  E0-=E0[0]
  plt.plot(E0,E,'bo')
  plt.plot(E0,E0,'g')
  #plt.show()

def cost5j(t,tt,K,J,b):
  pass

def r25j(t,tt,K,J,b):
  pass

def plot5j(t,tt,K,J,b):
  pass

################################################################################3
#BOTH ENERGY AND BANDS: 3 STATES
def cost3_both(t,tt,K,J,a1,a2,a3,b,we,lim):
  print(t,tt,K,J)
  return cost3(t,tt,K,a1,a2,a3,lim)+we*cost3j(t,tt,K,J,b)

#BOTH ENERGY AND BANDS: 5 STATES
def cost5_both(t,tt,K,J,a1,a2,a3,b,we,lim):
  print(t,tt,K,J)
  return cost5(t,tt,K,a1,a2,a3,lim)+we*cost5j(t,tt,K,J,b)

################################################################################3
#SINGLE STATE BAND RUN
'''
state="FLP2"
res=scipy.optimize.minimize(lambda p: cost(p[0],p[1],p[2],p[3],state),(1.1,0.4,0.0,1.0))
t,tt,K,a=res.x
print(res.x)
print(r2(t,tt,K,a,state))
plot(t,tt,K,a,state)
'''

################################################################################3
#THREE STATE BAND RUN
'''
lim=10.
res=scipy.optimize.minimize(lambda p: cost3(p[0],p[1],p[2],p[3],p[4],p[5],lim),(1.1,0.4,0.0,1.0,1.0,1.0))
t,tt,K,a1,a2,a3=res.x
print(res.x)
print(r23(t,tt,K,a1,a2,a3))
plot(t,tt,K,a1,"FLP2")
plot(t,tt,K,a2,"FLP0")
plot(t,tt,K,a3,"BCOL2")
'''

#BAND ONLY CALCULATIONS
#lim=10, R2=0.91
#1.01,0.41,-0.25

#lim=0.5, R2=0.93
#0.93,0.38,-0.35

#lim=0.25, R2=0.905
#0.89, 0.40, -0.30

#lim=0.1, R2=0.857
#1.11, 0.42, -0.01


################################################################################3
#FIVE STATE BAND RUN


################################################################################3
#THREE STATE ENERGY RUN
'''
constraints=({'type':'eq','fun':lambda p: p[3]-0.18})
res=scipy.optimize.minimize(lambda p: cost3j(p[0],p[1],p[2],p[3],p[4]),(1.1,0.4,0.0,0.18,0.0),constraints=constraints)
t,tt,K,J,b=res.x
print(res.x)
print(r23j(t,tt,K,J,b))
'''

#BEST ENERGY, NO PRIOR; R2 = 1.0
#t,tt,K,J,b=1.95683621 , 1.07842494 , 0.01187048  ,0.42430482 , 0.
#plot3j(t,tt,K,J,b)
#plot(t,tt,K,2.7,"FLP2")
#plot(t,tt,K,1.7,"FLP0")
#plot(t,tt,K,2.0,"BCOL2")

#ENERGY, J=0.18 PRIOR; R2 = 0.9999
#t,tt,K,J,b=[ 2.09991319,  1.186827 ,  -0.53138732 , 0.18       , 0. ]
#plot3j(t,tt,K,J,b)
#plot(t,tt,K,2.7,"FLP2")
#plot(t,tt,K,1.7,"FLP0")
#plot(t,tt,K,2.0,"BCOL2")

################################################################################3
#FIVE STATE ENERGY RUN

################################################################################3
#THREE STATE ENERGY AND BAND RUN 
we=10000
lim=0.50
constraints=({'type':'eq','fun':lambda p: p[3]-0.18})
res=scipy.optimize.minimize(lambda p: cost3_both(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],we,lim),(1.1,0.4,0.0,0.18,1.0,1.0,1.0,0.0),constraints=constraints)
t,tt,K,J,a1,a2,a3,b=res.x
print(res.x)
print(r23(t,tt,K,a1,a2,a3),r23j(t,tt,K,J,b))
plt.subplot(221)
plot3j(t,tt,K,J,b)
plt.subplot(222)
plot(t,tt,K,a1,"FLP2")
plt.subplot(223)
plot(t,tt,K,a2,"FLP0")
plt.subplot(224)
plot(t,tt,K,a3,"BCOL2")
plt.show()


  #BOTH, we=500, lim=0.25, no prior 
  #[ 1.6342942   1.04500227 -0.10898317  0.24578019  2.03724719  1.36617593
  #  2.1072249   0.        ]
  #-0.0411191890527 0.961782253372

#BOTH, we=500, lim=0.25, prior 
#[ 1.54317204  0.99949966 -0.2051253   0.18        1.86787416  1.29499058
#1.92797765  0.        ]
#0.142822468874 0.934255787037

  #BOTH, we=500, lim=0.50, no prior 
  #[ 1.01764485  0.52582506 -0.20305853  0.12317816  1.1447826   0.68796536
  #  1.14936977  0.        ]
  #0.820632569637 0.486374951578

  #BOTH, we=500, lim=0.50, prior 
  #[ 0.97317782  0.59309789  0.02808442  0.18        1.17895892  0.59978638
  #  1.20947884  0.        ]
  #0.672633948027 0.700948414995

  #BOTH, we=1000, lim=0.25, no prior 
  #[ 1.86903497  1.0148674  -0.49110414  0.16861351  2.13329576  1.63380854
  #  2.15911293  0.        ]
  #0.136152879999 0.93762447945

  #BOTH, we=1000, lim=0.25, prior
  #[ 1.82636897  1.17514638 -0.33628852  0.18        2.15836916  1.62519548
  #  2.27871131  0.        ]
  #-0.289724721637 0.993160184377

  #BOTH, we=1000, lim=0.50, no prior 
  #[ 1.17036482  0.72750281 -0.20812057  0.11210407  1.3687195   0.82360796
  #  1.35724939  0.        ]
  #0.599797668433 0.815702022179

  #BOTH, we=1000, lim=0.50, prior 
  #[ 1.09513229  0.66067644 -0.0369862   0.18        1.32880423  0.70221732
  #  1.34213221  0.        ]
  #0.602529088828 0.792014833165

  #BOTH, we=5000, lim=0.25, no prior 
  #[ 1.78634064  1.10718792 -0.14234818  0.26931877  2.20097857  1.4936815
  #  2.28943498  0.        ]
  #-0.256003486874 0.998845101354
  
  #BOTH, we=5000, lim=0.25, prior 
  #[ 1.83859074  1.18171605 -0.33421339  0.18        2.19781239  1.66646885
  #  2.28921339  0.        ]
  #-0.288263866366 0.99585079083

  #BOTH, we=5000, lim=0.50, no prior 
  #[ 1.47177194  0.89601441 -0.33787705  0.11939003  1.72052863  1.10322048
  #  1.71012419  0.        ]
  #0.307550968657 0.94847850985

#BOTH, we=5000, lim=0.50, prior 
#[ 1.44849999  0.88271113 -0.19802328  0.18        1.7458344   1.02242011
#  1.76279801  0.        ]
#0.230646512052 0.950615709567

  #BOTH, we=10000, lim=0.25, no prior 
  #[ 2.016071    1.12957188 -0.431509    0.21502847  2.35377031  1.78969088
  #  2.40388205  0.        ]
  #-0.213875244969 0.998521916192

  #BOTH, we=10000, lim=0.25, prior 
  #[ 2.03047296  1.14477956 -0.50429255  0.18        2.35318904  1.8368022
  #  2.39342532  0.        ]
  #-0.199448873367 0.998399370019

  #BOTH, we=10000, lim=0.5, no prior 
  #[ 1.55639226  0.92741029 -0.1285855   0.24172635  1.9228615   1.06025698
  #  1.94144157  0.        ]
  #-0.011992055478 0.98012364139

  #BOTH, we=10000, lim=0.5, prior 
  #[ 1.57977776  0.95641728 -0.26024371  0.18        1.90545024  1.14302396
  #  1.91645421  0.        ]
  #0.053453936565 0.98088723129

################################################################################3
#FIVE STATE ENERGY AND BAND RUN 
