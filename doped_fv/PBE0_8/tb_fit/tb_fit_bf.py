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

def flp2(K,d):
  up=np.zeros((8,8))
  dn=np.array([[0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,d-2*K,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,-2*K,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,-2*K,0],
               [0,0,0,0,0,0,0,d-2*K]])
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
  elif(state=="FLP2"):
    H[0]+=flp2(K,d)[0]
    H[1]+=flp2(K,d)[1]
  H[0]+=NN(t,x,y)+NNN(tt,x,y)
  H[1]+=NN(t,x,y)+NNN(tt,x,y)
  wu,vr=np.linalg.eigh(H[0])
  wd,vr=np.linalg.eigh(H[1])
  return [wu,wd]

#State choice
state="BCOL0"

#Brute force fitting
d=10
r2=[]
parms=[]
for t in np.arange(0.7,1.5,0.1):
  for tt in np.arange(0.1,0.5,0.1):
    for K in np.arange(0.1,0.7,0.1):
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

      #Plot PBE0 bands and TB bands
      dic=json.load(open("pbe0_bands.p","r"))
      e=np.array(e)
      eu=np.array(eu)

      if(state=="COL0"):  
        sum2=0
        for i in range(3):
          u=eu[:,i]-eu[0,0]+dic['col0u'][1][0]-dic['col0u'][i+1][:len(eu)]
          dn=e[:,i]-eu[0,0]+dic['col0u'][1][0]-dic['col0u'][i+1][:len(e)]
          sum2+=np.dot(u,u)+np.dot(dn,dn)
        sum2/=6
        r2.append(sum2)
        parms.append((t,tt,K,d))
      elif(state=="BCOL0"):
        sum2=0
        for i in range(1):
          u=eu[:,i]-eu[0,0]+dic['bcol0u'][1][0]-dic['bcol0u'][i+1][:len(eu)]
          dn=e[:,i]-eu[0,0]+dic['bcol0u'][1][0]-dic['bcol0u'][i+1][:len(e)]
          sum2+=np.dot(u,u)+np.dot(dn,dn)
        sum2/=2
        r2.append(sum2)
        parms.append((t,tt,K,d))
      elif(state=="CHK0"):
        sum2=0
        for i in range(3):
          u=eu[:,i]-eu[0,0]+dic['chk0u'][1][0]-dic['chk0u'][i+1][:len(eu)]
          dn=e[:,i]-eu[0,0]+dic['chk0u'][1][0]-dic['chk0u'][i+1][:len(e)]
          sum2+=np.dot(u,u)+np.dot(dn,dn)
        sum2/=6
        r2.append(sum2)
        parms.append((t,tt,K,d))
      elif(state=="BLK0"):
        sum2=0
        for i in range(3):
          u=eu[:,i]-eu[0,0]+dic['blk0u'][1][0]-dic['blk0u'][i+1][:len(eu)]
          dn=e[:,i]-eu[0,0]+dic['blk0u'][1][0]-dic['blk0u'][i+1][:len(e)]
          sum2+=np.dot(u,u)+np.dot(dn,dn)
        sum2/=6
        r2.append(sum2)
        parms.append((t,tt,K,d))
      elif(state=="FLP2"):
        sum2=0
        for i in range(3):
          dn=e[:,i]-e[0,0]+dic['flp2d'][1][0]-dic['flp2d'][i+1][:len(e)]
          sum2+=np.dot(dn,dn)
        sum2/=3
        r2.append(sum2)
        parms.append((t,tt,K,d))

ind=np.argsort(r2)
print("BEST")
print(r2[ind[0]])
print(parms[ind[0]])
