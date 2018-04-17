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

#Calculate TB bands
if(state=="COL0"):                #0.00 eV
  #8.53822272496
  t,tt,K,d=(1.2, 0.3, 0.3, 10)
elif(state=="FLP2"):              #0.10 eV
  #5.41337724077
  t,tt,K,d=(1.0, 0.4, 0.3, 10)
elif(state=="BLK0"):              #0.25 eV
  #4.09972034692
  t,tt,K,d=(1.0, 0.4, 0.3, 10)
elif(state=="CHK0"):              #0.50 eV
  #10.5415018513
  t,tt,K,d=(1.1, 0.4, 0.3, 10)
elif(state=="BCOL0"):             #1.00 eV
  #2.08149348526
  t,tt,K,d=(1.1, 0.4, 0.3, 10)

#############################################
#AVERAGE: 1.08(7), 0.38(4), 0.30(0), 10 
##############################################

#Brute force fitting
'''
d=10
r2=[]
parms=[]
for t in [1.0,1.1,1.2]:
  for tt in [0.2,0.3,0.4]:
    for K in [0.1,0.2,0.3]:
'''
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
  '''
  sum2=0
  for i in range(2):
    u=eu[:,i]-eu[0,0]+dic['col0u'][1][0]-dic['col0u'][i+1][0]
    dn=e[:,i]-eu[0,0]+dic['col0u'][1][0]-dic['col0u'][i+1][0]
    sum2+=np.dot(u,u)+np.dot(dn,dn)
  sum2/=4
  r2.append(sum2)
  parms.append((t,tt,K,d))
  '''
  for i in range(1,len(dic['col0u'])):
    plt.plot(dic['col0u'][i][:len(eu)],'k-')
    plt.plot(dic['col0u'][i][:len(e)],'r-')
  for i in range(4):
    plt.plot(eu[:,i]-eu[0,0]+dic['col0u'][1][0],'k-o')
    plt.plot(e[:,i]-eu[0,0]+dic['col0u'][1][0],'r-o')
elif(state=="BCOL0"):
  '''
  sum2=0
  for i in range(1):
    u=eu[:,i]-eu[0,0]+dic['bcol0u'][1][0]-dic['bcol0u'][i+1][0]
    dn=e[:,i]-eu[0,0]+dic['bcol0u'][1][0]-dic['bcol0u'][i+1][0]
    sum2+=np.dot(u,u)+np.dot(dn,dn)
  sum2/=2
  r2.append(sum2)
  parms.append((t,tt,K,d))
  '''
  for i in range(1,len(dic['bcol0u'])):
    plt.plot(dic['bcol0u'][i][:len(eu)],'k-')
    plt.plot(dic['bcol0u'][i][:len(e)],'r-')
  for i in range(4):
    plt.plot(eu[:,i]-eu[0,0]+dic['bcol0u'][1][0],'k-o')
    plt.plot(e[:,i]-eu[0,0]+dic['bcol0u'][1][0],'r-o')
elif(state=="CHK0"):
  '''
  sum2=0
  for i in range(2):
    u=eu[:,i]-eu[0,0]+dic['chk0u'][1][0]-dic['chk0u'][i+1][0]
    dn=e[:,i]-eu[0,0]+dic['chk0u'][1][0]-dic['chk0u'][i+1][0]
    sum2+=np.dot(u,u)+np.dot(dn,dn)
  sum2/=4
  r2.append(sum2)
  parms.append((t,tt,K,d))
  '''
  for i in range(1,len(dic['chk0u'])):
    plt.plot(dic['chk0u'][i][:len(eu)],'k-')
    plt.plot(dic['chk0u'][i][:len(e)],'r-')
  for i in range(4):
    plt.plot(eu[:,i]-eu[0,0]+dic['chk0u'][1][0],'k-o')
    plt.plot(e[:,i]-eu[0,0]+dic['chk0u'][1][0],'r-o')
elif(state=="BLK0"):
  '''
  sum2=0
  for i in range(2):
    u=eu[:,i]-eu[0,0]+dic['blk0u'][1][0]-dic['blk0u'][i+1][0]
    dn=e[:,i]-eu[0,0]+dic['blk0u'][1][0]-dic['blk0u'][i+1][0]
    sum2+=np.dot(u,u)+np.dot(dn,dn)
  sum2/=4
  r2.append(sum2)
  parms.append((t,tt,K,d))
  '''
  for i in range(1,len(dic['blk0u'])):
    plt.plot(dic['blk0u'][i][:len(eu)],'k-')
    plt.plot(dic['blk0u'][i][:len(e)],'r-')
  for i in range(4):
    plt.plot(eu[:,i]-eu[0,0]+dic['blk0u'][1][0],'k-o')
    plt.plot(e[:,i]-eu[0,0]+dic['blk0u'][1][0],'r-o')
elif(state=="FLP2"):
  '''
  sum2=0
  for i in range(3):
    dn=e[:,i]-eu[0,0]+dic['flp2d'][1][0]-dic['flp2d'][i+1][0]
    sum2+=np.dot(dn,dn)
  sum2/=3
  r2.append(sum2)
  parms.append((t,tt,K,d))
  '''
  for i in range(1,len(dic['flp2d'])):
    plt.plot(dic['flp2d'][i][:len(eu)],'r-')
  for i in range(4):
    plt.plot(e[:,i]-e[0,0]+dic['flp2d'][1][0],'r-o')
'''
ind=np.argsort(r2)
print("BEST")
for i in range(5):
  print(r2[ind[i]])
  print(parms[ind[i]])
'''
plt.axvline(11)
plt.axvline(11+11)
plt.axvline(11+11+15)
plt.axhline(0)
plt.title("PBE0, x=0.25, "+state)
plt.ylabel("E - EF (eV)")
plt.xticks([0,11,22,37],["(0,0)","(0,pi)","(pi,pi)","(0,0)"])
plt.text(3,4,"t="+str(t)+", tt="+str(tt)+"\nK="+str(K)+", d="+str(d))
plt.show()
