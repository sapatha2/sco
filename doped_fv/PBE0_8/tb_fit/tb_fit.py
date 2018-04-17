import numpy as np 
from numpy import exp 
import matplotlib.pyplot as plt
import json

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

'''
def NNold(t,x,y):
  M=-t*np.array([[0,1+exp(-1j*x),1+exp(1j*y),0],[0,0,0,1+exp(1j*y)],[0,0,0,1+exp(-1j*x)],[0,0,0,0]])
  M+=M.conj().T
  return M

#(0,0,0)[0,3] = (0,0,0)[0,1]
print("CHECK 1")
w,vr=np.linalg.eigh(NN(1.0,0,0))
wo,vro=np.linalg.eigh(NNold(1.0,0,0))
print(w[0],w[3])
print(wo[0],w[1])

print("CHECK 2")
wo,vro=np.linalg.eigh(NNold(1.0,np.pi,np.pi))
print(w[1],w[2])
print(wo[0],w[1])

print("CHECK 3")
w,vr=np.linalg.eigh(NN(1.0,np.pi,0))
wo,vro=np.linalg.eigh(NNold(1.0,np.pi/2,np.pi/2))
print(w[0],w[1])
print(wo[0])

print("CHECK 4")
print(w[2],w[3])
print(wo[1])
'''

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

def bcol0(K,d):
  up=np.array([[d,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,d,0,0,0,0],
               [0,0,0,0,d,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,d]])
  dn=np.array([[0,0,0,0,0,0,0,0],
               [0,d,0,0,0,0,0,0],
               [0,0,d,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,d,0,0],
               [0,0,0,0,0,0,d,0],
               [0,0,0,0,0,0,0,0]])
  return [up,dn]

def chk0(K,d):
  up=np.array([[2*K,0,0,0,0,0,0,0],
               [0,2*K,0,0,0,0,0,0],
               [0,0,d-2*K,0,0,0,0,0],
               [0,0,0,2*K,0,0,0,0],
               [0,0,0,0,d-2*K,0,0,0],
               [0,0,0,0,0,2*K,0,0],
               [0,0,0,0,0,0,d-2*K,0],
               [0,0,0,0,0,0,0,d-2*K]])
  dn=np.array([[d-2*K,0,0,0,0,0,0,0],
               [0,d-2*K,0,0,0,0,0,0],
               [0,0,2*K,0,0,0,0,0],
               [0,0,0,d-2*K,0,0,0,0],
               [0,0,0,0,2*K,0,0,0],
               [0,0,0,0,0,d-2*K,0,0],
               [0,0,0,0,0,0,2*K,0],
               [0,0,0,0,0,0,0,2*K]])
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

def tb(t,tt,K,d,x,y,state):
  H=[np.zeros((8,8))+0j,np.zeros((8,8))+0j]
  if(state=="COL0"):  
    H[0]+=col0(K,d)[0]
    H[1]+=col0(K,d)[1]
  elif(state=="BCOL0"):
    H[0]+=bcol0(K,d)[0]
    H[1]+=bcol0(K,d)[1]
  elif(state=="CHK0"):
    H[0]+=chk0(K,d)[0]
    H[1]+=chk0(K,d)[1]
  elif(state=="BLK0"):
    H[0]+=blk0(K,d)[0]
    H[1]+=blk0(K,d)[1]
  H[0]+=NN(t,x,y)+NNN(tt,x,y)
  H[1]+=NN(t,x,y)+NNN(tt,x,y)
  wu,vr=np.linalg.eigh(H[0])
  wd,vr=np.linalg.eigh(H[1])
  return [wu,wd]

#State choice
state="BLK0"

#Calculate TB bands
t,tt,K,d=(0.8,0.3,0.0,10)

e=[]
eu=[]
N=12
x=np.linspace(0,np.pi,N)
y=np.linspace(0,np.pi,N)
for i in range(N-1):
  eu.append(tb(t,tt,K,d,x[0],y[i],state)[0])
  e.append(tb(t,tt,K,d,x[0],y[i],state)[1])
N=12
x=np.linspace(0,np.pi,N)
y=np.linspace(0,np.pi,N)
for i in range(N-1):
  eu.append(tb(t,tt,K,d,x[i],y[N-1],state)[0])
  e.append(tb(t,tt,K,d,x[i],y[N-1],state)[1])
N=16
x=np.linspace(0,np.pi,N)
y=np.linspace(0,np.pi,N)
for i in range(N):
  eu.append(tb(t,tt,K,d,x[N-1-i],y[N-1-i],state)[0])
  e.append(tb(t,tt,K,d,x[N-1-i],y[N-1-i],state)[1])

#Plot PBE0 and TB bands
e=np.array(e)
eu=np.array(eu)

for i in range(2):
  plt.plot(eu[:,i],'k')
  plt.plot(e[:,i],'r')

plt.axvline(11)
plt.axvline(11+11)
plt.axvline(11+11+15)
plt.title("t="+str(t)+", tt="+str(tt)+", K="+str(K)+", d="+str(d))
plt.show()
