import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt

plt.suptitle("ED results for 12 site AFM chain (w/ PBC)")

N=12
pp=0

M=np.zeros((2**N,2**N))
for i in range(2**N):
  b=format(i, '0'+str(N)+'b')
  for k in range(len(b)):
    M[i,i]-=0.25
    x=list(b)
    if(k<len(b)-1):
      x[k],x[k+1]=x[k+1],x[k]
    else:
      x[k],x[0]=x[0],x[k]
    x=''.join(x)
    j=int(x,2)
    M[i,j]+=0.5

w,vr=eigh(M)
kk=np.argsort(w)

#Plot of weights, shows major contributors and fluctuations
plt.subplot(241)
plt.title("Ground state coeff")
plt.ylabel("Coeff")
plt.xlabel("basis element, binary")
plt.plot(np.arange(2.0**N),vr[:,pp],'.') #N=12, should be at 1365 and 2730; N=8, 170 and 85

#Calculate spin density on site 
#With fluctuations
states=np.arange(2**N)
states=[format(x,'0'+str(N)+'b') for x in states]
sd=np.zeros(N)
for s in range(N):
  for p in range(2**N):
    sd[s]+=abs(int(states[p][s])-0.5)*vr[p,pp]**2

plt.subplot(242)
plt.title("Local spin density w/ fluct")
plt.xlabel("Site")
plt.plot(sd,'og')

#Without fluctuations
states=np.arange(2**N)
states=[format(x,'0'+str(N)+'b') for x in states]
sd=np.zeros(N)
m=max(vr[:,0]**2)
for s in range(N):
  for p in range(2**N):
    if(vr[p,pp]**2==m): 
      sd[s]+=abs(int(states[p][s])-0.5)

plt.subplot(246)
plt.title("Local spin density w/o fluct")
plt.xlabel("Site")
plt.plot(sd,'or')

#Spin spin correlation
#With fluctutations
ss=np.zeros(N)
for s in range(N):
  M=np.zeros((2**N,2**N))
  for i in range(2**N):
    b=format(i, '0'+str(N)+'b')
    M[i,i]=(int(b[0])-1./2)*(int(b[0+s])-1./2)
  ss[s]=np.dot(np.transpose(vr[:,pp]),np.dot(M,vr[:,pp]))

plt.subplot(243)
plt.title("Sz0 Szj spin spin corr, w/ fluct")
plt.xlabel("Site")
plt.plot(ss,'og')

plt.subplot(244)
ss=np.zeros((N,N))
for k in range(N):
  for s in range(N):
    M=np.zeros((2**N,2**N))
    for i in range(2**N):
      b=format(i, '0'+str(N)+'b')
      M[i,i]=(int(b[k])-1./2)*(int(b[s])-1./2)
    ss[k,s]=np.dot(np.transpose(vr[:,pp]),np.dot(M,vr[:,pp]))

plt.subplot(244)
MQ=np.zeros(N+1)+0j
for n in range(N+1):
  for k in range(N):
    for p in range(N):
      MQ[n]+=np.exp(1j*(n*np.pi/N)*(k-p))*ss[k,p]
MQ/=max(MQ)
plt.plot(np.arange(N+1)/N,np.real(MQ),'og')
plt.plot(np.arange(N+1)/N,np.imag(MQ),'sg')
plt.title("M(Q)^2/max(M(Q)^2)")
plt.xlabel("Q/pi")

#Without fluctutations
ss=np.zeros(N)
for s in range(N):
  M=np.zeros((2**N,2**N))
  for i in range(2**N):
    if(vr[i,pp]**2==m):
      b=format(i, '0'+str(N)+'b')
      M[i,i]=(int(b[0])-1./2)*(int(b[0+s])-1./2)
  ss[s]=np.dot(np.transpose(vr[:,pp]),np.dot(M,vr[:,pp]))/(m)

plt.subplot(247)
plt.title("Sz0 Szj corr, w/out fluct")
plt.xlabel("Site")
plt.plot(ss,'or')

plt.subplot(248)
ss=np.zeros((N,N))
for k in range(N):
  for s in range(N):
    M=np.zeros((2**N,2**N))
    for i in range(2**N):
      if(vr[i,pp]**2==m):
        b=format(i, '0'+str(N)+'b')
        M[i,i]=(int(b[k])-1./2)*(int(b[s])-1./2)
    ss[k,s]=np.dot(np.transpose(vr[:,pp]),np.dot(M,vr[:,pp]))/(m)

MQ=np.zeros(N+1)+0j
for n in range(N+1):
  for k in range(N):
    for p in range(N):
      MQ[n]+=np.exp(1j*(n*np.pi/N)*(k-p))*ss[k,p]
MQ/=max(MQ)
plt.plot(np.arange(N+1)/N,np.real(MQ),'or')
plt.plot(np.arange(N+1)/N,np.imag(MQ),'sr')
plt.title("M(Q)^2/max(M(Q)^2)")
plt.xlabel("Q/pi")

plt.show()
