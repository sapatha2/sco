import numpy as np 
from numpy import exp 
import matplotlib.pyplot as plt

def NN(t,x,y):
  M=-t*np.array([[0,1+exp(-1j*x),1+exp(1j*y),0],[0,0,0,1+exp(1j*y)],[0,0,0,1+exp(-1j*x)],[0,0,0,0]])
  M+=M.conj().T
  return M

def NNN(tt,x,y):
  M=tt*np.array([[0,0,0,1+exp(-1j*x)+exp(1j*y)+exp(1j*(-x+y))],[0,0,1+exp(1j*x)+exp(1j*y)+exp(1j*(x+y)),0],[0,0,0,0],[0,0,0,0]])
  M+=M.conj().T
  return M

def flp1e(t,tt,K,d,x,y):
  return np.array([[d-2*K,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,-2*K]])+0j

def fme(t,tt,K,d,x,y):
  return np.zeros((4,4))+0j

def flp3e(t,tt,K,d,x,y):
  return np.array([[d,0,0,0],[0,d+2*K,0,0],[0,0,2*K,0],[0,0,0,d]])+0j

def tb(t,tt,K,d,x,y,n):
  H=np.zeros((4,4))+0j
  if(n==0):
    H=flp1e(t,tt,K,d,x,y)
  elif(n==1):
    H=cole(t,tt,K,d,x,y)
  elif(n==2):
    H=fme(t,tt,K,d,x,y)
  else:
    H=flp3e(t,tt,K,d,x,y)
  
  H+=NN(t,x,y)
  H+=NNN(tt,x,y)
  w,vr=np.linalg.eigh(H)
  return w

t=0.0
tt=0.0
K=0.0
d=0.0
n=0

if(n==0):
  #t,tt,K,d=(0.8,0.0,0.14,10)
  t,tt,K,d=(0.95,0.3,0.14,10)
elif(n==1):
  t,tt,K,d=(0.65,0.0,0.14,10)
elif(n==2):
  #t,tt,K,d=(0.6,0.0,0.14,10)
  t,tt,K,d=(0.875,0.25,0.14,10)
else:
  #t,tt,K,d
  pass
  
e=[]
N=14
x=np.linspace(0,np.pi,N)
y=np.linspace(0,np.pi,N)
for i in range(N):
  e.append(tb(t,tt,K,d,x[0],y[i],n))
N=13
x=np.linspace(0,np.pi,N)
y=np.linspace(0,np.pi,N)
for i in range(N):
  e.append(tb(t,tt,K,d,x[i],y[N-1],n))
N=18
x=np.linspace(0,np.pi,N)
y=np.linspace(0,np.pi,N)
for i in range(N):
  e.append(tb(t,tt,K,d,x[N-1-i],y[N-1-i],n))

e=np.array(e)
for i in range(np.shape(e)[1]):
  plt.plot(e[:,i])
plt.show()

