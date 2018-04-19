#Systematic fitting
import numpy as np
import json
import matplotlib.pyplot as plt
from tb_fit import tb 
from sigTB_calc import sigTB
import scipy.optimize

############
#PROBLEM - THE SCALE OF SEPARATION OF THE BINDING VERSUS THE ENERGIES IS DRASTICALLY DIFFERENT!
###########

##########################################################################################################
#PBE0 data
d0=[]

#PBE0 bands
bands=json.load(open("pbe0_bands.p","r"))

flpbands=(np.array(bands['FLP1']['down'][1][:45]+bands['FLP1']['down'][2][:45])-0.04215*27.2).tolist()
colbands=(np.array(bands['COL']['up'][2][:45]+bands['COL']['down'][2][:45])-0.04558*27.2).tolist()
fmbands=(np.array(bands['FM']['down'][0][:45]+bands['FM']['down'][1][:45])-0.04779*27.2).tolist()

#PBE0 energies 
E=np.array([-9.2092669022287E+02,-9.2091969201655E+02,-9.2091264054414E+02])
E-=E[0]
E*=27.2
E=E.tolist()

#Construct big vector
d0+=flpbands+colbands+fmbands
d0+=E

##########################################################################################################
#TB data

def dp(t,tt,K,J,m):
  ret=[] #Returned vector
  d=10  #fermi expulsion 

  #Calculate TB bands
  for n in range(3): #FLP, COL, CHK
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
    N=14
    x=np.linspace(0,np.pi,N)
    y=np.linspace(0,np.pi,N)
    for i in range(1,N):
      if(n!=1):
        e.append(tb(t,tt,K,d,x[i],y[N-1],n))
      else:
        eu.append(tb(t,tt,K,d,x[i],y[N-1],n,m)[0])
        e.append(tb(t,tt,K,d,x[i],y[N-1],n,m)[1])
    N=19
    x=np.linspace(0,np.pi,N)
    y=np.linspace(0,np.pi,N)
    for i in range(1,N):
      if(n!=1):
        e.append(tb(t,tt,K,d,x[N-1-i],y[N-1-i],n))
      else:
        eu.append(tb(t,tt,K,d,x[N-1-i],y[N-1-i],n,m)[0])
        e.append(tb(t,tt,K,d,x[N-1-i],y[N-1-i],n,m)[1])
   
    e=np.array(e)
    eu=np.array(eu)
    if(n!=1): ret+=(e[:,0]).tolist()+(e[:,1]).tolist()
    else:     ret+=(eu[:,0]).tolist()+(e[:,0]).tolist()
    
  #Calculate energies  
  sigJ=[0,0,2]
  tmp=[]
  for n in range(3): #FLP, COL, CHK
    ret+=[sigTB(t,tt,K,m,n)+J*sigJ[n]]
  
  return ret

##########################################################################################################
#Cost function
def cost(t,tt,K,J,m,we,a,b):
  dpp=dp(t,tt,K,J,m)
  w=np.identity(len(dpp))
  for i in range(len(dpp)-3,len(dpp)):
    w[i,i]=we
  delta=np.array(dpp)-np.array(d0)
  #Constants
  delta-=a
  delta[-3:]-=(b-a)
  #Number of points
  delta/=np.sqrt(len(delta[:-3]))
  delta[-3:]*=np.sqrt(len(delta[:-3])/len(delta[-3:]))
  #MSE on each part 
  #print(np.dot(delta[:-3],delta[:-3])/(we*np.dot(delta[-3:],delta[-3:])))
  return np.dot(delta,np.dot(w,delta))

'''
#Minimize cost function 
welist=[0.001,0.01,0.1,1,10,100,1000]
xlist=[]
for we in welist:
  print("########################################################################################")
  print("we="+str(we))
  res=scipy.optimize.minimize(lambda x: cost(x[0],x[1],x[2],x[3],x[4],we,x[5],x[6]), (1.0,0.3,0.3,0.3,0.7,0.0,-2.0))
  print("FINAL:",res.x)
  xlist.append(res.x)

print("########################################################################################")
print(welist)
print(xlist)
'''
##########################################################################################################
