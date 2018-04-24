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

def flp2(K,d):
  up=np.zeros((8,8))
  dn=np.array([[0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,d-K,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,-K,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,-K,0],
               [0,0,0,0,0,0,0,d-K]])
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

def tb(t,tt,K,d,x,y,state):
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
  '''
  elif(state=="BCOL0"):
    H[0]+=bcol0(K,d)[0]
    H[1]+=bcol0(K,d)[1]
  elif(state=="CHK0"):
    H[0]+=chk0(K,d)[0]
    H[1]+=chk0(K,d)[1]
  '''
  H[0]+=NN(t,x,y)+NNN(tt,x,y)
  H[1]+=NN(t,x,y)+NNN(tt,x,y)
  wu,vr=np.linalg.eigh(H[0])
  wd,vr=np.linalg.eigh(H[1])
  return [wu,wd]

'''
#########################################
#Minimization 
def dp(t,tt,K,J,state):
  d=10  #fermi expulsion 
  ret=[]
  #Calculate TB bands
  e=[]
  N=12
  x=np.linspace(0,np.pi,N)
  y=np.linspace(0,np.pi,N)
  for i in range(N):
      e.append(tb(t,tt,K,d,x[0],y[i],state)[1])
  N=12
  x=np.linspace(0,np.pi,N)
  y=np.linspace(0,np.pi,N)
  for i in range(1,N):
      e.append(tb(t,tt,K,d,x[i],y[N-1],state)[1])
  N=16
  x=np.linspace(0,np.pi,N)
  y=np.linspace(0,np.pi,N)
  for i in range(1,N):
      e.append(tb(t,tt,K,d,x[N-1-i],y[N-1-i],state)[1])
  e=np.array(e)
  ret+=(e[:,0]).tolist()+(e[:,1]).tolist()+(e[:,2]).tolist()
  return ret

#Single band optimization
def cost(t,tt,K,J,a,state):
  dpp=dp(t,tt,K,J,state)
  dic=json.load(open("pbe0_bands.p","r"))
  d0=dic[state.lower()+"d"][1][:38]+dic[state.lower()+"d"][2][:38]+dic[state.lower()+"d"][3][:38]
  delta=np.array(d0)-np.array(dpp)-a
  return np.dot(delta,delta)/len(delta)

import scipy.optimize
from sklearn.metrics import r2_score

def r2(t,tt,K,J,a,state):
  dpp=np.array(dp(t,tt,K,J,state))+a
  dic=json.load(open("pbe0_bands.p","r"))
  d0=dic[state.lower()+"d"][1][:38]+dic[state.lower()+"d"][2][:38]+dic[state.lower()+"d"][3][:38]
  return r2_score(d0,dpp)

for state in ["COL0","FLP2","BLK0","FLP0","BCOL2"]:
  res=scipy.optimize.minimize(lambda x: cost(x[0],x[1],x[2],x[3],x[4],state), (1.0,0.3,0.0,0.0,0.0))
  print(state+": ",r2(res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],state))
  print(res.x)
'''

'''
def cost(t,tt,K,J,a1,a2,a3,a4,a5):
  dic=json.load(open("pbe0_bands.p","r"))
  dpp=[]
  d0=[]
  for state in ["COL0","FLP2","BLK0","FLP0","BCOL2"]:
    dpp+=dp(t,tt,K,J,state)
    d0+=dic[state.lower()+"d"][1][:38]+dic[state.lower()+"d"][2][:38]+dic[state.lower()+"d"][3][:38]
  delta=np.array(d0)-np.array(dpp)
  consts=[a1,a2,a3,a4,a5]
  for i in range(5):
    delta[i*38*3:(i+1)*38*3]-=consts[i]

  return np.dot(delta,delta)/len(delta)

import scipy.optimize
from sklearn.metrics import r2_score

def r2(t,tt,K,J,a1,a2,a3,a4,a5):
  dic=json.load(open("pbe0_bands.p","r"))
  dpp=[]
  d0=[]
  for state in ["COL0","FLP2","BLK0","FLP0","BCOL2"]:
    dpp+=dp(t,tt,K,J,state)
    d0+=dic[state.lower()+"d"][1][:38]+dic[state.lower()+"d"][2][:38]+dic[state.lower()+"d"][3][:38]
 
  consts=[a1,a2,a3,a4,a5]
  for i in range(5):
    dpp[i*38*3:(i+1)*38*3]+=consts[i]

  return r2_score(d0,dpp)

res=scipy.optimize.minimize(lambda x: cost(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]), (1.0,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0))
print(r2(res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],res.x[5],res.x[6],res.x[7],res.x[8]))
print(res.x)
'''

'''
#SINGLE
COL0:  0.979823412231
[ 0.90866866  0.10983035  0.          0.          1.28182606]
FLP2:  0.952164003456
[ 1.0205741   0.63598913 -0.3104035   0.          1.00656838]
BLK0:  0.98603500388
[ 1.21055316  0.51830952  0.          0.          1.54462455]
FLP0:  0.987937125854
[ 0.84276737  0.39897574  0.12769619  0.          0.53869696]
BCOL2:  0.987036804989
[ 1.02777623  0.4095624  -0.40880138  0.          0.97155692]

#TOGETHER
[ 1.00994872  0.39794366 -0.29908201  0.          1.34818717  0.90529563
  1.30226456  0.42277115  0.98174547]

#E + e
 1.33185463  0.47671784 -0.01219605  0.49488093  1.65166881  1.46785919
 1.65448035  0.93375195  1.5358998   0.00893655
'''
'''
#State choice
state="BCOL2"
#Calculate TB bands
d=10
if(state=="COL0"):                #0.00 eV, GOOD
  t,tt,K,J,a=[1.33185463,  0.47671784, -0.01219605,  0.49488093,  1.65166881]
elif(state=="FLP2"):              #0.10 eV, OK 
  t,tt,K,J,a=[1.33185463,  0.47671784, -0.01219605,  0.49488093,  1.46785919]
elif(state=="BLK0"):              #0.25 eV, OK 
  t,tt,K,J,a=[1.33185463,  0.47671784, -0.01219605,  0.49488093,  1.65448035]
elif(state=="FLP0"):              #0.10 eV, OK 
  t,tt,K,J,a=[1.33185463,  0.47671784, -0.01219605,  0.49488093,   0.93375195]
elif(state=="BCOL2"):              #0.10 eV, OK 
  t,tt,K,J,a=[1.33185463,  0.47671784, -0.01219605,  0.49488093,  1.5358998]
elif(state=="CHK0"):              #0.50 eV, GOOD
  #10.5415018513
  t,tt,K,d=(0.69999999999999996, 0.40000000000000002, 0.10000000000000001, 10)
elif(state=="BCOL0"):             #1.00 eV, OK
  #2.08149348526
  t,tt,K,d=(1.0999999999999999, 0.40000000000000002, 0.59999999999999998, 10)

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
e=np.array(e)+a
eu=np.array(eu)

if(state=="COL0"):  
  for i in range(1,len(dic['col0d'])):
    plt.plot(dic['col0d'][i][:len(e)],'r-')
  for i in range(4):
    plt.plot(e[:,i],'r-o')
elif(state=="FLP2"):
  for i in range(1,len(dic['flp2d'])):
    plt.plot(dic['flp2d'][i][:len(eu)],'r-')
  for i in range(4):
    plt.plot(e[:,i],'r-o')
elif(state=="BLK0"):
  for i in range(1,len(dic['blk0d'])):
    plt.plot(dic['blk0d'][i][:len(eu)],'k-')
  for i in range(4):
    plt.plot(e[:,i],'r-o')
elif(state=="FLP0"):
  for i in range(1,len(dic['flp0d'])):
    plt.plot(dic['flp0d'][i][:len(e)],'r-')
  for i in range(4):
    plt.plot(e[:,i],'r-o')
elif(state=="BCOL2"):
  for i in range(1,len(dic['bcol2d'])):
    plt.plot(dic['bcol2d'][i][:len(e)],'r-')
  for i in range(4):
    plt.plot(e[:,i],'r-o')
'''
'''
elif(state=="BCOL0"):
  for i in range(1,len(dic['bcol0u'])):
    plt.plot(dic['bcol0u'][i][:len(eu)],'k-')
    plt.plot(dic['bcol0u'][i][:len(e)],'r-')
  for i in range(4):
    plt.plot(eu[:,i]-eu[0,0]+dic['bcol0u'][1][0],'k-o')
    plt.plot(e[:,i]-eu[0,0]+dic['bcol0u'][1][0],'r-o')
elif(state=="CHK0"):
  for i in range(1,len(dic['chk0u'])):
    plt.plot(dic['chk0u'][i][:len(eu)],'b-o')
    plt.plot(dic['chk0u'][i][:len(e)],'g-o')
  for i in range(3):
    plt.plot(eu[:,i]-eu[0,0]+dic['chk0u'][1][0],'k-o')
    plt.plot(e[:,i]-eu[0,0]+dic['chk0u'][1][0],'r-o')
'''
'''
plt.axvline(11)
plt.axvline(11+11)
plt.axvline(11+11+15)
plt.axhline(0)
plt.title("PBE0, x=0.25, "+state)
plt.ylabel("E - EF (eV)")
plt.xticks([0,11,22,37],["(0,0)","(0,pi)","(pi,pi)","(0,0)"])
plt.text(3,1.25,"t="+str(t)+", tt="+str(tt)+"\nK="+str(K)+", d="+str(d))
plt.show()
'''
'''
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
'''
