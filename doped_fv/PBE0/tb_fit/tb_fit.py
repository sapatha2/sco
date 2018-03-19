import numpy as np 
from numpy import exp 
import matplotlib.pyplot as plt
import json

def NN(t,x,y):
  M=-t*np.array([[0,1+exp(-1j*x),1+exp(1j*y),0],[0,0,0,1+exp(1j*y)],[0,0,0,1+exp(-1j*x)],[0,0,0,0]])
  M+=M.conj().T
  return M

def NNN(tt,x,y):
  M=tt*np.array([[0,0,0,1+exp(-1j*x)+exp(1j*y)+exp(1j*(-x+y))],[0,0,1+exp(1j*x)+exp(1j*y)+exp(1j*(x+y)),0],[0,0,0,0],[0,0,0,0]])
  M+=M.conj().T
  return M

def chke(t,tt,K,d,x,y):
  return np.array([[d-2*K,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,-2*K]])+0j

def colue(t,tt,K,d,m,x,y):
  return np.array([[d-m,0,0,0],[0,d-m,0,0],[0,0,-m,0],[0,0,0,-m]])+0j

def cole(t,tt,K,d,x,y):
  return np.array([[0,0,0,0],[0,0,0,0],[0,0,d,0],[0,0,0,d]])+0j

def fme(t,tt,K,d,x,y):
  return np.zeros((4,4))+0j

def flp3e(t,tt,K,d,x,y):
  return np.array([[d,0,0,0],[0,d+2*K,0,0],[0,0,2*K,0],[0,0,0,d]])+0j

def tb(t,tt,K,d,x,y,n,m=0):
  Hu=np.zeros((4,4))+0j
  H=np.zeros((4,4))+0j
  if(n==0):
    H=chke(t,tt,K,d,x,y)
  elif(n==1):
    Hu=colue(t,tt,K,d,m,x,y)
    H=cole(t,tt,K,d,x,y)
  elif(n==2):
    H=fme(t,tt,K,d,x,y)
  else:
    H=flp3e(t,tt,K,d,x,y)
 
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

#State choice
n=0

#Read in PBE0 band structure
di={}
with open("pbe0_bands.p","r") as inp:
  di=json.load(inp)

m=0
#Parameter unpacking
if(n==0):
  t,tt,K,d=(0.95,0.3,0.3,10)
elif(n==1):
  t,tt,K,d,m=(0.80,0.3,0.3,10,0.7)
elif(n==2):
  t,tt,K,d=(0.875,0.25,0.3,10)
else:
  t,tt,K,d=(0.80,0.3,0.3,1)
  pass
  
#Calculate TB bands
e=[]
eu=[]
N=14
x=np.linspace(0,np.pi,N)
y=np.linspace(0,np.pi,N)
for i in range(N):
  if(n!=1):
    e.append(tb(t,tt,K,d,x[0],y[i],n))
  else:
    eu.append(tb(t,tt,K,d,x[0],y[i],n,m)[0])
    e.append(tb(t,tt,K,d,x[0],y[i],n,m)[1])
N=13
x=np.linspace(0,np.pi,N)
y=np.linspace(0,np.pi,N)
for i in range(N):
  if(n!=1):
    e.append(tb(t,tt,K,d,x[i],y[N-1],n))
  else:
    eu.append(tb(t,tt,K,d,x[i],y[N-1],n,m)[0])
    e.append(tb(t,tt,K,d,x[i],y[N-1],n,m)[1])
N=18
x=np.linspace(0,np.pi,N)
y=np.linspace(0,np.pi,N)
for i in range(N):
  if(n!=1):
    e.append(tb(t,tt,K,d,x[N-1-i],y[N-1-i],n))
  else:
    eu.append(tb(t,tt,K,d,x[N-1-i],y[N-1-i],n,m)[0])
    e.append(tb(t,tt,K,d,x[N-1-i],y[N-1-i],n,m)[1])

#Plot PBE0 and TB bands
e=np.array(e)
eu=np.array(eu)
print(np.shape(e))
print(np.shape(eu))
if(n<0):
  plt.plot(di['FLP1']['down'][1][:45],'r')
  plt.plot(di['FLP1']['down'][2][:45],'r')
  plt.plot(di['COL']['up'][2][:45],'k--')
  plt.plot(di['COL']['down'][2][:45],'r--')
  plt.plot(di['FM']['down'][0][:45],'r.-')
  plt.plot(di['FM']['down'][1][:45],'r.-')
elif(n==0):
  #Bands: 466, 467
  c=['b','g','m']
  for i in range(np.shape(e)[1]-2):
    plt.plot(e[:,i]-e[0,0]+di['FLP1']['down'][1][0],c[i])
  for i in range(5):
    plt.plot(di['FLP1']['up'][i][:45],'k')
    plt.plot(di['FLP1']['down'][i][:45],'r')
elif(n==1):
  #Bands: 67, 467 
  c=['b','g']
  for i in range(np.shape(eu)[1]-2):
    plt.plot(eu[:,i]-e[0,0]+di['COL']['down'][2][0],c[i])
  c=['b--','g--']
  for i in range(np.shape(e)[1]-2):
    plt.plot(e[:,i]-e[0,0]+di['COL']['down'][2][0],c[i])
  for i in range(5):
    plt.plot(di['COL']['up'][i][:45],'k')
    plt.plot(di['COL']['down'][i][:45],'r')
elif(n==2):
  #Bands: 465, 466
  c=['b','g','m']
  for i in range(np.shape(e)[1]-1):
    plt.plot(e[:,i]-e[0,0]+di['FM']['down'][0][0],c[i])
  for i in range(5):
    plt.plot(di['FM']['up'][i][:45],'k')
    plt.plot(di['FM']['down'][i][:45],'r')
else:
  c=['b','g','m']
  for i in range(np.shape(e)[1]-1):
    plt.plot(e[:,i],c[i])
  pass

plt.axhline(0.0,color='gray',linestyle="--")
plt.title("t="+str(t)+", tt="+str(tt)+", K="+str(K)+", d="+str(d))
plt.show()

