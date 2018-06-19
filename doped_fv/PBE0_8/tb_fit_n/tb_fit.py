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
  ret+=cost(t,tt,K,a1,"COL0",lim)
  ret+=cost(t,tt,K,a2,"FLP2",lim)
  ret+=cost(t,tt,K,a3,"FLP0",lim)
  ret+=cost(t,tt,K,a4,"BCOL2",lim)
  return ret

def r24(t,tt,K,a1,a2,a3,a4):
  d0=[]
  dpp=[]
  bands=json.load(open("pbe0_bands.p","r"))
  for state in ["COL0","FLP2","FLP0","BCOL2"]:
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
  return r2_score(d0,dpp) 

def plot4(t,tt,K,a1,a2,a3,a4):
  j=0
  const=[a1,a2,a3,a4]
  for state in ["COL0","FLP2","FLP0","BCOL2"]:
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
  for state in ["COL0","FLP2","FLP0","BCOL2"]:
    E.append(sigTB(t,tt,K,state)+J*sigJ[ind])
    ind+=1
  E=np.array(E)
  E+=b
  E-=E[0]

  E0=np.array([0,0.106,0.252,0.285])
  E0-=E0[0]

  return np.dot(E0-E,E0-E)

def r24j(t,tt,K,J,b):
  sigJ=[0,0,-2,0]
  E=[]
  ind=0
  for state in ["COL0","FLP2","FLP0","BCOL2"]:
    E.append(sigTB(t,tt,K,state)+J*sigJ[ind])
    ind+=1
  E=np.array(E)
  E+=b
  E-=E[0]
  
  E0=np.array([0,0.106,0.252,0.285])
  E0-=E0[0]
  
  return r2_score(E0,E)

def plot4j(t,tt,K,J,b):
  sigJ=[0,0,-2,0]
  E=[]
  ind=0
  for state in ["COL0","FLP2","FLP0","BCOL2"]:
    E.append(sigTB(t,tt,K,state)+J*sigJ[ind])
    ind+=1
  E=np.array(E)
  E+=b
  E-=E[0]
  
  E0=np.array([0,0.106,0.252,0.285])
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
state="flp0_n"
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
lim=0.10
res=scipy.optimize.minimize(lambda p: cost4(p[0],p[1],p[2],p[3],p[4],p[5],p[6],lim),(1.1,0.4,0.0,1.0,1.0,1.0,1.0))
t,tt,K,a1,a2,a3,a4=res.x
print(res.x)
print(r24(t,tt,K,a1,a2,a3,a4))
plot4(t,tt,K,a1,a2,a3,a4)
plt.show()
'''

#lim=10
#[ 1.03309165  0.41679968 -0.28269269  1.39277115  0.98451076  0.72912682
#  1.01296482]
#0.979107614569

#lim=0.5, GOOD
#[ 0.95423733  0.39311385 -0.36783962  1.32334485  0.94950138  0.69532971
#  0.91285412]
#0.966425980952

#lim=0.25, GOOD
#[ 0.9448341   0.41690881 -0.39270645  1.32437299  0.95799299  0.71801252
#  0.92647914]
#0.954748836004

#lim=0.10
#[  1.10003627e+00   4.00000000e-01   3.62721859e-05   1.00000000e+00
#   1.00000000e+00   1.00000000e+00   1.00000000e+00]
#0.776011504656

################################################################################3
#FIVE STATE BAND RUN
'''
lim=0.25
res=scipy.optimize.minimize(lambda p: cost5(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],lim),(1.1,0.4,0.0,1.0,1.0,1.0,1.0,1.0))
t,tt,K,a1,a2,a3,a4,a5=res.x
print(res.x)
print(r25(t,tt,K,a1,a2,a3,a4,a5))
plot5(t,tt,K,a1,a2,a3,a4,a5)
plt.show()
'''

#lim=10 
#[ 1.04870376  0.38667457 -0.38235622  1.41909932  0.9680397   0.76550659
#  0.98487879  1.96470275]
#0.951342368501

#lim=0.5, GOOD
#[ 0.98538601  0.35686016 -0.39523105  1.3854178   0.91088625  0.69960854
#  0.85591214  1.90389896]
#0.944264066146

#lim=0.25, GOOD
#[ 0.50804435  0.46979156  0.70091488  0.69413234  1.01582272  0.1557477
#  0.98888813  1.16061958]
#0.753670912628

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
constraints=({'type':'eq','fun':lambda p: p[3]-0.18})
res=scipy.optimize.minimize(lambda p: cost4j(p[0],p[1],p[2],p[3],p[4]),(2.0,1.0,-0.5,0.18,0.0),constraints=constraints)
#res=scipy.optimize.minimize(lambda p: cost4j(p[0],p[1],p[2],p[3],p[4]),(2.0,1.0,0.01,0.40,0.0))
t,tt,K,J,b=res.x
print(res.x)
print(r24j(t,tt,K,J,b))
plot4j(t,tt,K,J,b)
plt.show()
'''

#BEST ENERGY, NO PRIOR; R2=1.0
#t,tt,K,J,b=[ 1.99316482,  1.10664349 ,-0.15530005 , 0.34752776  ,0.        ]
#plot4(t,tt,K,3.,2.8,1.8,2.5)
#plt.show()

#BEST ENERGY, PRIOR; R2=0.999937
#t,tt,K,J,b=[ 1.88109837 , 1.19214456 ,-0.36226218 , 0.18   ,     0.        ]
#plot4(t,tt,K,2.5,2.3,1.8,2.5)
#plt.show()

################################################################################3
#FIVE STATE ENERGY RUN

'''
#constraints=({'type':'eq','fun':lambda p: p[3]-0.18})
#res=scipy.optimize.minimize(lambda p: cost5j(p[0],p[1],p[2],p[3],p[4]),(2.0,1.0,-0.5,0.18,0.0),constraints=constraints)
res=scipy.optimize.minimize(lambda p: cost5j(p[0],p[1],p[2],p[3],p[4]),(2.0,1.0,0.01,0.40,0.0))
t,tt,K,J,b=res.x
print(res.x)
print(r25j(t,tt,K,J,b))
plot5j(t,tt,K,J,b)
plt.show()
'''

#BEST ENERGY, NO PRIOR; R2=0.99999999998
#t,tt,K,J,b=[ 2.66800418,  0.90816689,  0.25241026,  1.04962781,  0.        ]
#plot5(t,tt,K,3.5,3.2,3.,3.,3.)
#plt.show()

#BEST ENERGY, PRIOR; R2=0.998334499389; OK except for BLK0
#t,tt,K,J,b=[ 2.54663242 , 1.7116688 , -0.58922306,  0.18 ,       0.        ]
#plot5(t,tt,K,3.5,3.2,3.,3.,3.)

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
constraints=({'type':'eq','fun':lambda p: p[3]-0.18})
res=scipy.optimize.minimize(lambda p: cost4_both(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],we,lim),(2.0,1.0,-0.5,0.18,1.0,1.0,1.0,1.0,0.0),constraints=constraints)
t,tt,K,J,a1,a2,a3,a4,b=res.x
print(res.x)
'''
'''
#t,tt,K,J,a1,a2,a3,a4,b=[1.49904204 , 0.89523692 ,-0.24776747 , 0.18      ,  2.08147531 , 1.79489747, 1.07267895 , 1.80422129 , 0.]
t,tt,K,J,a1,a2,a3,a4,b=[1.03536369,  0.53250351 ,-0.15009942,  0.18  ,      1.42954778,  1.18583609, 0.67371943,  1.19444543,  0.]
print(r24(t,tt,K,a1,a2,a3,a4),r24j(t,tt,K,J,b))
plot4(t,tt,K,a1,a2,a3,a4)
plt.subplot(236)
plot4j(t,tt,K,J,b)
plt.show()
'''
#####################################################
#WITH COL0, NOT BLK0

  #BOTH we=500, lim=0.25, prior 
  #[ 1.68266524  1.04065727 -0.30037414  0.18        2.33745977  2.01049345
  #1.45388352  2.04033842  0.        ]
  #0.447928418641 0.998051886998

#BOTH we=500, lim=0.50, prior 
#[ 0.89865065  0.41910265 -0.13252855  0.18        1.22478048  0.98173056
#0.57221688  1.00156728  0.        ]
#0.951829602783 0.863014765666 

#BOTH we=1000, lim=0.25, prior 
#[ 1.17630512  0.64056284 -0.17154732  0.18        1.65203126  1.36779188
#  0.85985089  1.40100511  0.        ]
#0.894187643723 0.953195130158

#BOTH we=1000, lim=0.50, prior 
#[ 1.03536369  0.53250351 -0.15009942  0.18        1.42954778  1.18583609
#  0.67371943  1.19444543  0.        ]
#0.939456821809 0.915352293528

  #BOTH we=5000, lim=0.25, prior 
  #[ 1.75709487  1.09545354 -0.32152392  0.18        2.44344329  2.09028037
  #1.54806359  2.14075407  0.        ]
  #0.354150659492 0.999668187252

#BOTH we=5000, lim=0.50, prior 
#[ 1.27731418  0.72162831 -0.19216477  0.18        1.77146028  1.51013717
#  0.87234738  1.52440621  0.        ]
#0.829868587921 0.973007646232

  #BOTH we=10000, lim=0.25, prior 
  #[ 1.77740249  1.11136486 -0.32846669  0.18        2.46886568  2.12189603
  #1.568913    2.15844412  0.        ]
  #0.325523556925 0.999777199124

#BOTH we=10000, lim=0.50, prior 
#[ 1.49904204  0.89523692 -0.24776747  0.18        2.08147531  1.79489747
#  1.07267895  1.80422129  0.        ]
#0.636712023352 0.993108294582


#####################################################
#WITH BLK0

#BOTH we=500, lim=0.25, prior 
#[ 1.61838761  0.85384637 -0.38252938  0.18        2.17446736  1.8411622
#  1.33501015  1.86236135  0.        ]
#0.665942931102 0.803823497137

#BOTH we=500, lim=0.50, prior 
#[ 0.99478098  0.45926059 -0.13284148  0.18        1.25999894  1.10471784
#  0.62196026  1.11342153  0.        ]
#0.952109141612 0.248021808276

#BOTH we=1000, lim=0.25, prior 
#[ 1.815914    0.97351816 -0.45697051  0.18        2.48307548  2.07615505
#  1.56062396  2.09258934  0.        ]
#0.47840894953 0.899380581302

#BOTH we=1000, lim=0.50, prior 
#[ 1.11611251  0.54198956 -0.18015184  0.18        1.42781346  1.27622251
#  0.72221874  1.26267111  0.        ]
#0.924782615388 0.40197702278

#BOTH we=5000, lim=0.25, prior 
#[ 2.15569686  1.18643912 -0.57822218  0.18        3.03515043  2.47987916
#  1.98513957  2.51185104  0.        ]
#0.0402272199531 0.991945583565

#BOTH we=5000, lim=0.50, prior 
#[ 1.72407388  0.92554537 -0.42220798  0.18        2.31313918  2.03695664
#  1.27504879  1.97242502  0.        ]
#0.523842400444 0.867818148041

#BOTH we=10000, lim=0.25, prior 
#[ 2.21899681  1.2272606  -0.59917772  0.18        3.14795739  2.55452059
#  2.06731885  2.59152052  0.        ]
#-0.0590539503041 0.99794253774

#BOTH we=10000, lim=0.50, prior 
#[ 1.92852648  1.05020724 -0.49780701  0.18        2.62150292  2.28848403
#  1.47124129  2.2164197   0.        ]
#0.284043129274 0.946681854983


################################################################################3
#FIVE STATE ENERGY AND BAND RUN 
'''
we=5000
lim=0.25

constraints=({'type':'ineq','fun':lambda p: p[3]-0.17},{'type':'ineq','fun':lambda p: 0.19-p[3]})
res=scipy.optimize.minimize(lambda p: cost5_both(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],we,lim),(1.0,0.4,-0.2,0.18,1.0,1.0,1.0,1.0,1.0,0.0),constraints=constraints)
t,tt,K,J,a1,a2,a3,a4,a5,b=res.x
print(res.x)
print(r25(t,tt,K,a1,a2,a3,a4,a5),r25j(t,tt,K,J,b))
plot5(t,tt,K,a1,a2,a3,a4,a5)
plt.subplot(236)
plot5j(t,tt,K,J,b)
plt.show()
'''

#BOTH we=500, lim=0.25, prior 
#[ 0.84557427  0.37985557 -0.13263135  0.18        1.19648626  0.91665028
#  1.06256679  0.55239987  0.95618958  0.        ]
#0.937647475972 0.853052618739

#BOTH we=500, lim=0.50, prior 
#[ 0.85620977  0.37851787 -0.12818159  0.18        1.16214647  0.90619292
#  1.06892965  0.53954525  0.92620135  0.        ]
#0.948243742627 0.843616968055

#BOTH we=1000, lim=0.25, prior 
#[ 0.84768663  0.38064388 -0.12891803  0.18        1.16003428  0.90408865
#  1.07375465  0.52797397  0.96186965  0.        ]
#0.940779567948 0.853517838276

#BOTH we=1000, lim=0.5, prior
#[ 0.86129897  0.39005201 -0.1239589   0.18        1.15322378  0.91660654
#  1.0858662   0.5598753   0.93570781  0.        ]
#0.946790654819 0.852894727904

#BOTH we=5000, lim=0.25, prior 
#[ 2.5181844   1.69191875 -0.58448285  0.18        3.49391034  2.96840026
#  4.39576162  2.53478593  3.20241377  0.        ]
#-1.14882258093 0.997588684854

#BOTH we=5000, lim=0.50, prior 
#[ 0.85271808  0.38588732 -0.12837876  0.18        1.1606117   0.91258828
#  1.07381732  0.54040829  0.93057374  0.        ]
#0.943813835323 0.854316837424

#BOTH we=10000, lim=0.50, prior 
#[ 2.42865202  1.62895845 -0.56550819  0.18        3.39332277  2.94700078
#  3.8893102   2.01122989  2.99801548  0.        ]
#-0.929852196261 0.985977286891

#BOTH we=10000, lim=1.0, prior 
#[ 0.86550789  0.395143   -0.12817725  0.18        1.15588578  0.83733542
#  1.08124921  0.55484747  0.88518376  0.        ]
#0.950951514996 0.853777901381

#BOTH we=50000, lim=0.25, prior 
#[ 2.54442688  1.7102568  -0.58893157  0.18        3.55175178  2.99981698
#  4.44151228  2.5648478   3.23776836  0.        ]
#-1.20572023601 0.998329105281

#BOTH we=50000, lim=0.50, prior 
#[ 2.52272559  1.69503356 -0.58464383  0.18        3.52764298  3.07038074
#  4.06171499  2.10482096  3.12051282  0.        ]
#-1.15048402719 0.997816223066

#BOTH we=5000, lim=0.25, prior vary
#[ 0.85688344  0.38664905 -0.12500135  0.18332143  1.21313526  0.92215724
#  1.08274591  0.54720519  0.93970739  0.        ]
#0.943741725492 0.854225284524

################################################################################3
#PARETO PLOTS

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
pareto4=[[0.447928418641, 0.998051886998],
[0.951829602783, 0.863014765666],
[0.894187643723, 0.953195130158],
[0.939456821809, 0.915352293528],
[0.354150659492, 0.999668187252],
[0.829868587921, 0.973007646232],
[0.325523556925, 0.999777199124],
[0.636712023352, 0.993108294582],]
pareto4=np.array(pareto4)

pareto4_=[[0.665942931102, 0.803823497137],
[0.952109141612, 0.248021808276],
[0.47840894953, 0.899380581302],
[0.924782615388, 0.40197702278],
[0.0402272199531, 0.991945583565],
[0.523842400444, 0.867818148041],
[-0.0590539503041, 0.99794253774],
[0.284043129274, 0.946681854983]]
pareto4_=np.array(pareto4_)

pareto5=[[0.937647475972, 0.853052618739],
[0.948243742627, 0.843616968055],
[0.940779567948, 0.853517838276],
[0.946790654819, 0.852894727904],
[-1.14882258093, 0.997588684854],
[0.943813835323, 0.854316837424],
[-0.929852196261, 0.985977286891],
[0.950951514996, 0.853777901381],
[-1.20572023601, 0.998329105281],
[-1.15048402719, 0.997816223066]]
pareto5=np.array(pareto5)
'''

plt.plot(pareto3[:,0],pareto3[:,1],'go')
plt.plot(0.9371337025509947, 0.9997876553183775,'bs')
plt.title("3-state Pareto")
plt.xlabel("R^2 band")
plt.ylabel("R^2 energy")
plt.show()
