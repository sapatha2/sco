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

#############################################
#AVERAGE: 1.08(7), 0.38(4), 0.30(0), 10 
##############################################

col0sum=[]
flp2sum=[]
blk0sum=[]
chk0sum=[]
bcol0sum=[]
#for N in [10,20,30,50]:
for N in [10]:
  for state in ["COL0","FLP2","BLK0","CHK0","BCOL0"]:
    #Calculate TB bands
    if(state=="COL0"):                #0.00 eV, GOOD
      #8.53822272496
      t,tt,K,d=(0.99999999999999989, 0.40000000000000001, 0.5, 10)
    elif(state=="FLP2"):              #0.10 eV, OK 
      #5.41337724077
      t,tt,K,d=(1.0999999999999999, 0.40000000000000002, 0.10000000000000001, 10)
    elif(state=="BLK0"):              #0.25 eV, OK 
      #4.09972034692
      t,tt,K,d=(0.99999999999999989, 0.40000000000000002, 0.59999999999999998, 10)
    elif(state=="CHK0"):              #0.50 eV, GOOD
      #10.5415018513
      t,tt,K,d=(0.69999999999999996, 0.40000000000000002, 0.10000000000000001, 10)
    elif(state=="BCOL0"):             #1.00 eV, OK
      #2.08149348526
      t,tt,K,d=(1.0999999999999999, 0.40000000000000002, 0.59999999999999998, 10)
    e=[]
    rho=[]
    x=np.linspace(-np.pi+2*np.pi/(N-1),np.pi,N-1)
    y=np.linspace(-np.pi+2*np.pi/(N-1),np.pi,N-1)

    for i in range(N-1):
      for j in range(N-1):
        if(state!="FLP2"):
          w=tb(t,tt,K,d,x[i],y[j],state)
          for k in range(len(w[0])):
            e.append(w[0][k])
            e.append(w[1][k])
        else:
          w=tb(t,tt,K,d,x[i],y[j],state)
          for k in range(len(w[0])):
            e.append(w[1][k])
    
    e=sorted(e)
    e=e[:2*(N-1)**2]
    e=np.sum(e)/((N-1)**2)
    if(state=="COL0"):                #0.00 eV
      col0sum.append(e)
    elif(state=="FLP2"):              #0.10 eV
      flp2sum.append(e)
    elif(state=="BLK0"):              #0.25 eV
      blk0sum.append(e)
    elif(state=="CHK0"):              #0.50 eV
      chk0sum.append(e)
    elif(state=="BCOL0"):             #1.00 eV
      bcol0sum.append(e)

print(col0sum)
print(flp2sum)
print(blk0sum)
print(chk0sum)
print(bcol0sum)
