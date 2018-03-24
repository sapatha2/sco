import numpy as np
from numpy import exp
import matplotlib.pyplot as plt

#Revelant functions
def NN(t,x,y):
  M=-t*np.array([[0,1+exp(-1j*x),1+exp(1j*y),0],[0,0,0,1+exp(1j*y)],[0,0,0,1+exp(-1j*x)],[0,0,0,0]])
  M+=M.conj().T
  return M

def NNN(tt,x,y):
  M=tt*np.array([[0,0,0,1+exp(-1j*x)+exp(1j*y)+exp(1j*(-x+y))],[0,0,1+exp(1j*x)+exp(1j*y)+exp(1j*(x+y)),0],[0,0,0,0],[0,0,0,0]])
  M+=M.conj().T
  return M

def chkue(K,d):
  return np.array([[2*K,0,0,0],[0,d,0,0],[0,0,d,0],[0,0,0,d+2*K]])+0j

def chke(K,d):
  return np.array([[d-2*K,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,-2*K]])+0j

def colue(K,d,m):
  return np.array([[d-m,0,0,0],[0,d-m,0,0],[0,0,-m,0],[0,0,0,-m]])+0j

def cole(K,d):
  return np.array([[0,0,0,0],[0,0,0,0],[0,0,d,0],[0,0,0,d]])+0j

def fme(K,d):
  return np.zeros((4,4))+0j

def tb(t,tt,K,d,x,y,n,m=0):
  Hu=np.zeros((4,4))+0j
  H=np.zeros((4,4))+0j
  if(n==0):
    H=chke(K,d)
  elif(n==1):
    Hu=colue(K,d,m)
    H=cole(K,d)
  elif(n==2):
    H=fme(K,d)
  else:
    pass

  H+=NN(t,x,y)
  H+=NNN(tt,x,y)
  if(n!=1):
    w,vr=np.linalg.eigh(H)
    return w
  else:
    Hu+=NN(t,x,y)
    Hu+=NNN(tt,x,y)
    wu,vru=np.linalg.eigh(Hu)
    w,vr=np.linalg.eigh(H)
    return [wu,w]

#State selection
flpsum=[]
colsum=[]
fmsum=[]

for N in [10,20,30,50]:
  for n in range(3):
    if(n==0):
      t,tt,K,d=(0.85,0.30,0.1,10)
    elif(n==1):
      t,tt,K,d,m=(0.75,0.20,0.1,10,0.20)
    elif(n==2):
      t,tt,K,d=(0.75,0.25,0.1,10)
    else:
      pass

    #Calculation
    e=[]
    rho=[]
    x=np.linspace(-np.pi+2*np.pi/(N-1),np.pi,N-1)
    y=np.linspace(-np.pi+2*np.pi/(N-1),np.pi,N-1)

    for i in range(N-1):
      for j in range(N-1):
        if(n!=1):
          e+=tb(t,tt,K,d,x[i],y[j],n)[:2].tolist()
        else:
          e.append(tb(t,tt,K,d,x[i],y[j],n,m)[0][0])
          e.append(tb(t,tt,K,d,x[i],y[j],n,m)[1][0])

    e=sorted(e)
    e=e[:(N-1)**2]
    e=np.sum(e)/(N-1)**2
    if(n==0): flpsum.append(e)
    elif(n==1): colsum.append(e)
    else: fmsum.append(e)

#Use these values in J_fit
print(flpsum)
print(colsum)
print(fmsum)

