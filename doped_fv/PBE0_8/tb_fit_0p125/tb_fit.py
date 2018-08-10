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

def chk(K,d):
  up=np.diag([d-K,d-K,K,d-K,K,d-K,K,K])
  dn=np.diag([K,K,d-K,K,d-K,K,d-K,d-K])
  return [up,dn]

def col(K,d):
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

def achn(K,d):
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

def blk(K,d):
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
def flp(K,d):
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

def diag(K,d):
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

def tb(t,tt,K,x,y,state,m=0):
  d=10
  H=[np.zeros((8,8))+0j,np.zeros((8,8))+0j]
  if(state=="col1"):  
    H[0]+=col(K,d)[0]
    H[1]+=col(K,d)[1]
  elif(state=="flp1"):
    H[0]+=flp(K,d)[0]
    H[1]+=flp(K,d)[1]
  elif(state=="chk1"):
    H[0]+=chk(K,d)[0]
    H[1]+=chk(K,d)[1]
  elif(state=="achn3"):
    H[0]+=achn(K,d)[0]
    H[1]+=achn(K,d)[1]
  else:
    print("tb not defined for this state")
    exit(0)
  H[0]+=NN(t,x,y)+NNN(tt,x,y)
  H[1]+=NN(t,x,y)+NNN(tt,x,y)+m
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

  e=np.array(e)+a
  ret=[]
  for i in range(2):
    ret+=e[:,i].tolist()
  return np.array(ret)

def cost(t,tt,K,a,state,lim):
  bands=json.load(open("pbe0_bands.p","r"))
  d0=[]
  for i in range(2):
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
  for i in range(2):
    d0+=bands[state.lower()][i][:38]
  d0=np.array(d0)
  dpp=dp(t,tt,K,a,state)
  return r2_score(d0,dpp) 

def plot(t,tt,K,a,state):
  bands=json.load(open("pbe0_bands.p","r"))
  d0=[]
  for i in range(2):
    d0+=bands[state.lower()][i][:38]
  d0=np.array(d0)
  dpp=dp(t,tt,K,a,state)
  
  #plt.ylabel("E-Ef, eV")
  plt.plot(d0,'go')
  plt.plot(dpp,'b.')
  plt.ylim([-1.0,1.5])
  plt.yticks(np.arange(-1.0,2.0,1.0))
  plt.xticks(np.arange(1.0,1.0,1.0))
  plt.axhline(0,color='gray',linestyle='--')

def cost3(t,tt,K,a1,a2,a3,lim):
  ret=0
  ret+=cost(t,tt,K,a1,"flp1",lim)
  ret+=cost(t,tt,K,a2,"chk1",lim)
  ret+=cost(t,tt,K,a3,"achn3",lim)
  return ret

def r23(t,tt,K,a1,a2,a3):
  d0=[]
  dpp=[]
  for state in ["flp1","chk1","achn3"]:
    bands=json.load(open("pbe0_bands.p","r"))
    for i in range(2):
      d0+=bands[state.lower()][i][:38]
  
  dpp+=dp(t,tt,K,a1,"flp1").tolist()
  dpp+=dp(t,tt,K,a2,"chk1").tolist()
  dpp+=dp(t,tt,K,a3,"achn3").tolist()
  return r2_score(d0,dpp) 

def cost4(t,tt,K,a1,a2,a3,a4,lim):
  ret=0
  ret+=cost(t,tt,K,a1,"flp1",lim)
  ret+=cost(t,tt,K,a2,"chk1",lim)
  ret+=cost(t,tt,K,a3,"achn3",lim)
  ret+=cost(t,tt,K,a4,"col1",lim)
  return ret

def r24(t,tt,K,a1,a2,a3,a4):
  d0=[]
  dpp=[]
  bands=json.load(open("pbe0_bands.p","r"))
  for state in ["flp1","chk1","achn3","col1"]:
    for i in range(2):
      d0+=bands[state.lower()][i][:38]

  dpp+=dp(t,tt,K,a1,"flp1").tolist()
  dpp+=dp(t,tt,K,a2,"chk1").tolist()
  dpp+=dp(t,tt,K,a3,"achn3").tolist()
  dpp+=dp(t,tt,K,a4,"col1").tolist()
  return r2_score(d0,dpp) 

def plot4(t,tt,K,a1,a2,a3,a4):
  j=0
  const=[a1,a2,a3,a4]
  for state in ["flp1","chk1","achn3","col1"]:
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
def sigTB(t,tt,K,state,m):
  d=10
  N=50
  e=[]
  rho=[]
  x=np.linspace(-np.pi+2*np.pi/(N-1),np.pi,N-1)
  y=np.linspace(-np.pi+2*np.pi/(N-1),np.pi,N-1)

  for i in range(N-1):
    for j in range(N-1):
      w=tb(t,tt,K,x[i],y[j],state)
      if(state in ["chk1","col1"]):
        w=tb(t,tt,K,x[i],y[j],state,m=m)
        for k in range(len(w[0])):
          e.append(w[0][k])
          e.append(w[1][k])
      else:
        for k in range(len(w[1])):
          e.append(w[1][k])

  e=sorted(e)
  #plt.plot(e,'b.')
  e=e[:(N-1)**2]
  #plt.plot(e,'g.')
  #plt.show()
  return np.sum(e)/((N-1)**2)

def cost3j(t,tt,K,J,b,m):
  print(t,tt,K,J,m)
  sigJ=[-2,-4,0]
  E=[]
  ind=0
  for state in ["flp1","chk1","achn3"]:
    E.append(sigTB(t,tt,K,state,m)+J*sigJ[ind])
    ind+=1
  E=np.array(E)
  E+=b
  E-=E[0]
  
  E0=np.array([0,0.049,0.149])
  E0-=E0[0]

  return np.dot(E0-E,E0-E)

def r23j(t,tt,K,J,b,m):
  sigJ=[-2,-4,0]
  E=[]
  ind=0
  for state in ["flp1","chk1","achn3"]:
    E.append(sigTB(t,tt,K,state,m)+J*sigJ[ind])
    ind+=1
  E=np.array(E)
  E+=b
  E-=E[0]
  
  E0=np.array([0,0.049,0.149])
  E0-=E0[0]
  
  return r2_score(E0,E)

def plot3j(t,tt,K,J,b,m):
  sigJ=[-2,-4,0]
  E=[]
  ind=0
  for state in ["flp1","chk1","achn3"]:
    E.append(sigTB(t,tt,K,state,m)+J*sigJ[ind])
    ind+=1
  E=np.array(E)
  E+=b
  E-=E[0]
  
  E0=np.array([0,0.049,0.149])
  E0-=E0[0]
  plt.plot(E0,E,'bo')
  plt.plot(E0,E0,'g')
  #plt.show()

def cost4j(t,tt,K,J,b,m):
  print(t,tt,K,J,m)
  sigJ=[-2,-4,0,0]
  E=[]
  ind=0
  for state in ["flp1","chk1","achn3","col1"]:
    E.append(sigTB(t,tt,K,state,m)+J*sigJ[ind])
    ind+=1
  E=np.array(E)
  E+=b
  E-=E[0]

  E0=np.array([0,0.049,0.149,0.221])
  E0-=E0[0]

  return np.dot(E0-E,E0-E)

def r24j(t,tt,K,J,b,m):
  sigJ=[-2,-4,0,0]
  E=[]
  ind=0
  for state in ["flp1","chk1","achn3","col1"]:
    E.append(sigTB(t,tt,K,state,m)+J*sigJ[ind])
    ind+=1
  E=np.array(E)
  E+=b
  E-=E[0]
  
  E0=np.array([0,0.049,0.149,0.221])
  E0-=E0[0]
  
  return r2_score(E0,E)

def plot4j(t,tt,K,J,b,m):
  sigJ=[-2,-4,0,0]
  E=[]
  ind=0
  for state in ["flp1","chk1","achn3","col1"]:
    E.append(sigTB(t,tt,K,state,m)+J*sigJ[ind])
    ind+=1
  E=np.array(E)
  E+=b
  E-=E[0]
  
  E0=np.array([0,0.049,0.149,0.221])
  E0-=E0[0]
  plt.plot(E0,E,'bo')
  plt.plot(E0,E0,'g')
  plt.xticks(np.arange(0,0.3,0.1))
  plt.yticks(np.arange(0,0.3,0.1))
  plt.ylabel("Predicted, eV")
  plt.xlabel("DFT, eV")
  plt.title("Total Energy")

################################################################################3
#BOTH ENERGY AND BANDS: 3 STATES
def cost3_both(t,tt,K,J,a1,a2,a3,b,m,we,lim):
  print(t,tt,K,J)
  return cost3(t,tt,K,a1,a2,a3,lim)+we*cost3j(t,tt,K,J,b,m)

#BOTH ENERGY AND BANDS: 4 STATES
def cost4_both(t,tt,K,J,a1,a2,a3,a4,b,m,we,lim):
  print(t,tt,K,J)
  return cost4(t,tt,K,a1,a2,a3,a4,lim)+we*cost4j(t,tt,K,J,b,m)

################################################################################3
#PLOT BANDS
'''
bands=json.load(open("pbe0_bands.p","r"))
plt.plot(bands['chk1'][0],'bo')
plt.plot(bands['chk1'][1],'bo')
#plt.plot(bands['chk1'][2],'bo')
plt.show()
plt.plot(bands['col1'][0],'bo')
plt.plot(bands['col1'][1],'bo')
#plt.plot(bands['col1'][2],'bo')
plt.show()
plt.plot(bands['flp1'][0],'bo')
plt.plot(bands['flp1'][1],'bo')
#plt.plot(bands['flp1'][2],'bo')
plt.show()
plt.plot(bands['achn3'][0],'bo')
plt.plot(bands['achn3'][1],'bo')
#plt.plot(bands['achn3'][2],'bo')
plt.show()
'''

################################################################################3
#SINGLE STATE BAND RUN
'''
from scipy.optimize import Bounds
lim=10
for state in ["flp1","chk1","achn3","col1"]:
  res=scipy.optimize.minimize(lambda p: cost(p[0],p[1],p[2],p[3],state,lim),(1.0,0.4,0.0,2.0))
  t,tt,K,a=res.x
  print(t,tt,K,a)
  print(r2(t,tt,K,a,state))
  plot(t,tt,K,a,state)
  plt.show()
'''

################################################################################3
#THREE STATE BAND RUN
'''
lim=0.5
res=scipy.optimize.minimize(lambda p: cost3(p[0],p[1],p[2],p[3],p[4],p[5],lim),(1.1,0.4,0.0,1.0,1.0,1.0))
t,tt,K,a1,a2,a3=res.x
t*=-1
print(t,tt,K,a1,a2,a3)
print(r23(t,tt,K,a1,a2,a3))
plot(t,tt,K,a1,"flp1")
plt.show()
plot(t,tt,K,a2,"chk1")
plt.show()
plot(t,tt,K,a3,"achn3")
plt.show()
'''

#BAND ONLY CALCULATIONS
#lim=10
#0.9757901244474414 0.4017741777796167 0.4182800410225736 1.4541589156080563 0.21837188053636938 1.8842502493309279
#0.9722618799911984

#lim=0.5
#1.0293861564061242 0.4624171900705294 0.6253331107121146 1.635089598284964 0.10576375248512045 2.093131787947487
#0.950648057821299

################################################################################3
#FOUR STATE BAND RUN
'''
lim=0.25
res=scipy.optimize.minimize(lambda p: cost4(p[0],p[1],p[2],p[3],p[4],p[5],p[6],lim),(1.1,0.4,0.6,1.0,1.0,1.0,1.0))
t,tt,K,a1,a2,a3,a4=res.x
print(res.x)
print(r24(t,tt,K,a1,a2,a3,a4))
plot4(t,tt,K,a1,a2,a3,a4)
plt.show()
'''

#lim=10
#[1.0171907  0.40819949 0.34775049 1.50666829 0.30545383 1.92348381
# 1.43459673]
#0.9810013182919124

#lim=0.5, GOOD
#[1.05323922 0.46885172 0.60196144 1.66890936 0.1431581  2.12359157
# 1.529816  ]
#0.9659122559275077

#lim=0.25, GOOD
# 0.99752944  0.50103597  0.87646278  1.73245596 -0.10175006  2.23720709
#  1.456227  ]
#0.9336288596365827

################################################################################3
#THREE STATE ENERGY RUN
'''
#constraints=({'type':'eq','fun':lambda p: p[3]-0.18})
#res=scipy.optimize.minimize(lambda p: cost3j(p[0],p[1],p[2],p[3],p[4],p[5]),(1.1,0.4,0.0,0.18,0.0,0.4),constraints=constraints)
res=scipy.optimize.minimize(lambda p: cost3j(p[0],p[1],p[2],p[3],p[4],p[5]),(1.1,0.4,0.0,0.18,0.0,0.4))
t,tt,K,J,b,m=res.x
print(res.x)
print(r23j(t,tt,K,J,b,m))
'''

#BEST ENERGY, NO PRIOR; R2 = 1.0
'''
t,tt,K,J,b,m=[1.08469884, 0.44943491, 0.06123835 ,0.21930152,  0.,         0.40036826]
print(r23j(t,tt,K,J,b,m))
plot3j(t,tt,K,J,b,m)
plt.show()
plot(t,tt,K,1.6,'flp1')
plt.show()
plot(t,tt,K,0.6,'chk1')
plt.show()
plot(t,tt,K,2.0,'achn3')
plt.show()
'''

#ENERGY, J=0.18 PRIOR; R2 = 0.9999
'''
t,tt,K,J,b,m=[ 0.87333634 , 0.30553179, -0.10546454,  0.18  ,      0. ,         0.40079229]
print(r23j(t,tt,K,J,b,m))
plot3j(t,tt,K,J,b,m)
plt.show()
plot(t,tt,K,1.4,'flp1')
plt.show()
plot(t,tt,K,0.6,'chk1')
plt.show()
plot(t,tt,K,1.5,'achn3')
plt.show()
'''

################################################################################3
#FOUR STATE ENERGY RUN
'''
#constraints=({'type':'eq','fun':lambda p: p[3]-0.18})
#res=scipy.optimize.minimize(lambda p: cost4j(p[0],p[1],p[2],p[3],p[4],p[5]),(1.1,0.4,0.0,0.18,0.0,0.4),constraints=constraints)
res=scipy.optimize.minimize(lambda p: cost4j(p[0],p[1],p[2],p[3],p[4],p[5]),(1.1,0.4,0.0,0.18,0.0,0.4))
t,tt,K,J,b,m=res.x
print(res.x)
print(r24j(t,tt,K,J,b,m))
'''

#BEST ENERGY, NO PRIOR; R2=1.0
'''
t,tt,K,J,b,m=[ 0.9819783,   0.30161734, -0.33936621,  0.16218388,  0. ,         0.40099556]
plot4(t,tt,K,1.5,1.1,1.7,1.5)
plt.show()
'''

#BEST ENERGY, PRIOR; R2=0.999937
'''
t,tt,K,J,b,m=[ 1.29909543 , 0.42282308, -0.45739033 , 0.18  ,      0. ,         0.40102375]
plot4(t,tt,K,2.,1.6,2.2,2.)
plt.show()
'''
################################################################################3
#THREE STATE ENERGY AND BAND RUN 
'''
we=70
lim=0.25
constraints=({'type':'eq','fun':lambda p: p[3]-0.18})
res=scipy.optimize.minimize(lambda p: cost3_both(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],we,lim),(1.1,0.4,0.0,0.18,1.0,1.0,1.0,0.4,0.0),constraints=constraints)
t,tt,K,J,a1,a2,a3,b,m=res.x
print(res.x)
print(r23(t,tt,K,a1,a2,a3),r23j(t,tt,K,J,b,m))
plt.subplot(221)
plot3j(t,tt,K,J,b,m)
plt.subplot(222)
plot(t,tt,K,a1,"flp1")
plt.subplot(223)
plot(t,tt,K,a2,"chk1")
plt.subplot(224)
plot(t,tt,K,a3,"achn3")
plt.show()
'''

#BOTH, we=70, lim=0.50, prior 
#[ 0.97885429  0.35933657  0.11670769  0.18        1.41024044  0.46189514
#  1.73761081  0.4        -0.23839525]
#0.9597134279962551 0.5024891270852967

#BOTH, we=70, lim=0.25, prior 
#[ 1.10892972  0.36769315 -0.29266255  0.18        1.64717453  0.91722339
#  1.8525524   0.4        -0.12677657]
#0.9427902940100733 0.9969014525647493

#BOTH, we=30, lim=0.50, prior 
#[ 1.0098111   0.38487927  0.20140527  0.18        1.49763143  0.42594437
#  1.83839458  0.39999856 -0.26168318]
#0.9682641712571002 0.14962607819281915

#BOTH, we=30, lim=0.25, prior 
#[ 0.82544103  0.34601023  0.4140037   0.18        1.28618139  0.1240923
#  1.66165019  0.4        -0.28788851]
#0.951968702921596 0.381701582008799

#BOTH, we=100, lim=0.50, prior 
#[ 1.11993832  0.38020098 -0.27266866  0.18        1.63678124  0.91619393
#  1.84673068  0.4        -0.1002682 ]
#0.9490073666233102 0.983195992282267

#BOTH, we=500, lim=0.25, prior 
#[ 1.10217621  0.36870418 -0.28672616  0.18        1.60270317  0.91287761
#  1.8036976   0.4         0.04621146]
#0.9448119825030344 0.9970349789056789

################################################################################3
#FOUR STATE ENERGY AND BAND RUN 
'''
we=1000000
lim=0.25
constraints=({'type':'eq','fun':lambda p: p[3]-0.18})
res=scipy.optimize.minimize(lambda p: cost4_both(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],we,lim),(1.1,0.4,0.0,0.18,1.0,1.0,1.0,1.0,0.4,0.0),constraints=constraints)
t,tt,K,J,a1,a2,a3,a4,b,m=res.x
print(res.x)
print(r24(t,tt,K,a1,a2,a3,a4),r24j(t,tt,K,J,b,m))
'''
t,tt,K,J,a1,a2,a3,a4,b,m=[ 1.15170394,  0.39218111, -0.29262262 , 0.18  ,      1.69178952 , 0.95802699,
1.89141888 , 1.71880588,  0.39999982 , 0.20994019]

plt.subplot(232)
plt.title("FLP")
plot(t,tt,K,a1,"flp1")
plt.subplot(233)
plt.title("CHK")
plot(t,tt,K,a2,"chk1")
plt.subplot(235)
plt.title("ACHN")
plot(t,tt,K,a3,"achn3")
plt.subplot(236)
plt.title("COL")
plot(t,tt,K,a4,"col1")
plt.suptitle("x=0.125 DFT fitting results")
plt.subplot(234)
plot4j(t,tt,K,J,b,m)

#BOTH, we=10, lim=0.50, prior 
#[ 1.11945322  0.39492157 -0.16104265  0.18        1.642794    0.82084797
#  1.88622549  1.65996393  0.4         0.07353788]
#0.9651141083919738 0.7073169242324674

#BOTH, we=10, lim=0.25, prior 
#[ 1.0269352   0.34825565 -0.16995759  0.18        1.5147111   0.75010273
#  1.75027443  1.53394102  0.4         0.05824087]
#0.960032630350126 0.8279230597675337

#BOTH, we=30, lim=0.50, prior 
#[ 1.14625703  0.39515543 -0.25631834  0.18        1.68391301  0.92231051
#  1.89737752  1.7076529   0.4         0.59166568]
#0.9596918440983537 0.9455725282367278

#BOTH, we=30, lim=0.25, prior 
#[ 1.08955814  0.36021657 -0.28336347  0.18        1.61882747  0.89596343
#  1.81837784  1.63065715  0.4         0.42396277]
#0.9569486556239821 0.963487492746432

#BOTH, we=100, lim=0.50, prior 
#[ 1.15170394  0.39218111 -0.29262262  0.18        1.69178952  0.95802699
#  1.89141888  1.71880588  0.39999982  0.20994019]
#0.9576849284811734 0.980739250289097

#BOTH, we=500, lim=0.50, prior 
#[ 1.18540689  0.39905764 -0.33439483  0.18        1.74417026  1.03077228
#  1.93517428  1.78084081  0.4         0.21174178]
#0.9517012681254474 0.9902553121300764

#BOTH, we=1000, lim=0.25, prior 
#[ 1.26105891  0.41238034 -0.42445261  0.18        1.90082629  1.14611024
#  2.07195075  1.91193613  0.4         0.64158433]
#0.9315897483358968 0.9994505556717954

#BOTH, we=5000, lim=0.25, prior 
#[ 1.28726927  0.41927693 -0.44782001  0.18        1.94134963  1.1866911
#  2.110134    1.9593872   0.4         0.25910148]
#0.9233784944280758 0.9999776193959066

#BOTH, we=10000, lim=0.25, prior 
#[ 1.28589005  0.41880894 -0.44681061  0.18        1.9431489   1.18414604
#  2.10824043  1.95295235  0.4         0.1665393 ]
#0.9244160040480747 0.9999920291329627

#BOTH, we=100000, lim=0.25, prior
#[ 1.29367126  0.4211317  -0.45305339  0.18        1.95534746  1.1956239
#  2.12015909  1.96428724  0.4         0.25483103]
#0.9224591843859415 0.9999999073662887

#BOTH, we=500000, lim=0.25, prior 
#[ 0.63428865  0.34996763  0.34580138  0.18        1.02219698  0.12286621
#  1.24551923  0.89472015  0.4        -0.04988519]
#0.8403975404237436 0.9999988729753257

################################################################################3
#PARETO PLOTS
'''
pareto4=[[0.447928418641, 0.998051886998],
[0.951829602783, 0.863014765666],
[0.894187643723, 0.953195130158],
[0.939456821809, 0.915352293528],
[0.354150659492, 0.999668187252],
[0.829868587921, 0.973007646232],
[0.325523556925, 0.999777199124],
[0.636712023352, 0.993108294582],]
pareto4=np.array(pareto4)
'''

pareto4=[[0.9651141083919738, 0.7073169242324674],
[0.960032630350126, 0.8279230597675337],
[0.9596918440983537, 0.9455725282367278],
[0.9569486556239821, 0.963487492746432],
[0.9576849284811734, 0.980739250289097],
[0.9517012681254474, 0.9902553121300764],
[0.9315897483358968, 0.9994505556717954],
[0.9233784944280758, 0.9999776193959066],
[0.9244160040480747, 0.9999920291329627],
[0.9224591843859415, 0.9999999073662887],
[0.8403975404237436, 0.9999988729753257]]
pareto4=np.array(pareto4)

plt.subplot(231)
plt.plot(pareto4[:,0],pareto4[:,1],'go')
plt.plot(0.9576849284811734, 0.980739250289097,'bs')
plt.title("4-state Pareto")
plt.xlabel("R^2 band")
plt.ylabel("R^2 energy")
plt.show()
