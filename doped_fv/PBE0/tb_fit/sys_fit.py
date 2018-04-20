#Systematic fitting
import numpy as np
import json
import matplotlib.pyplot as plt
from tb_fit import tb 
from sigTB_calc import sigTB
import scipy.optimize
from sklearn.metrics import r2_score

############
#PROBLEM - THE SCALE OF SEPARATION OF THE BINDING VERSUS THE ENERGIES IS DRASTICALLY DIFFERENT!
###########

##########################################################################################################
#PBE0 data
d0=[[],[]]

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
d0[0]=flpbands+colbands+fmbands
d0[1]=E

##########################################################################################################
#TB data

def dp(t,tt,K,J,m):
  ret=[[],[]] #Returned vector
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
    if(n!=1): ret[0]+=(e[:,0]).tolist()+(e[:,1]).tolist()
    else:     ret[0]+=(eu[:,0]).tolist()+(e[:,0]).tolist()
    
  #Calculate energies  
  sigJ=[0,0,2]
  tmp=[]
  for n in range(3): #FLP, COL, CHK
    tmp+=[sigTB(t,tt,K,m,n)+J*sigJ[n]]
  
  tmp=np.array(tmp)-tmp[0]
  ret[1]=tmp.tolist()

  return ret

##########################################################################################################
#Cost function
def cost(t,tt,K,J,m,we,a,b):
  dpp=dp(t,tt,K,J,m)
  dpp[0]=np.array(dpp[0])+a
  dpp[1]=np.array(dpp[1])+b

  delta=[[],[]]
  delta[0]=dpp[0]-d0[0]
  delta[1]=dpp[1]-d0[1]

  #DEBUG
  #print(np.dot(delta[0],delta[0])/len(delta[0]),np.dot(delta[1],delta[1])/len(delta[1]))

  return np.dot(delta[0],delta[0])/len(delta[0])+we*np.dot(delta[1],delta[1])/len(delta[1])

#R2 function
def r2(t,tt,K,J,m,a,b):
  dpp=dp(t,tt,K,J,m)
  return (r2_score(d0[0],dpp[0]), r2_score(d0[1],dpp[1]))


##########################################################################################################
#Run 
welist=[0.0625,0.125,0.25,0.5,1,2,4,8,16]
xlist=[]
r2list=[[],[]]

'''
for we in welist:
  print("########################################################################################")
  print("we="+str(we))
  res=scipy.optimize.minimize(lambda x: cost(x[0],x[1],x[2],x[3],x[4],we,x[5],x[6]), (1.0,0.3,0.3,0.3,0.7,0.0,-2.0))
  xlist.append(res.x) #t,tt,K,J,m,a,b
  t,tt,K,J,m,a,b=res.x
  r2s=r2(t,tt,K,J,m,a,b)
  r2list[0].append(r2s[0])
  r2list[1].append(r2s[1])
 
print(xlist)
print(r2list[0])
print(r2list[1])
'''

xlist=[[ 0.90380277,  0.32018907,  0.41959119,  0.05855655,  0.69540466, 0.00892276,  0.09214191], 
[ 0.90374898,  0.32040386,  0.42645169,  0.05731828,  0.68928501,0.00946633,  0.08572412], 
[ 0.90348822,  0.32060591,  0.43783478,  0.05517982,  0.67942348,0.01017611,  0.07508505], 
[ 0.90324795,  0.3210019 ,  0.45358457,  0.0522184 ,  0.6654719 ,0.01130047,  0.06020936], 
[ 0.9028267 ,  0.32133707,  0.47171027,  0.04872806,  0.64964335,0.01245128,  0.04304985], 
[ 0.90200069,  0.32125502,  0.48843966,  0.04535691,  0.63575204,0.01302343,  0.02728524], 
[ 0.90138347,  0.32135407,  0.50086193,  0.04255208,  0.62642983,0.01364334,  0.01583295], 
[ 0.901152  ,  0.3214671 ,  0.50842143,  0.04098906,  0.61993311,0.01409007,  0.00859442], 
[ 0.90102094,  0.32153178,  0.51269844,  0.04010282,  0.61624764,0.01434317,  0.00448917]]

r2list[0]=[0.96026497157304502, 0.96013300817941194, 0.95976742496514234, 0.9589317459157144, 0.95751272236358487, 
           0.95580197188629978, 0.95429925708925278, 0.95323249402820298, 0.95258902974425075]
r2list[1]=[0.41872570083663974, 0.49681250355993101, 0.61391465942448997, 0.75175954278617296, 0.87309726006162935, 
           0.94902307891831428, 0.98283470715932042, 0.99494212697384121, 0.99862006781047985]
'''
#Plot R2 comparison
plt.subplot(221)
plt.ylabel("R2 band")
plt.xlabel("WE")
plt.plot(welist,r2list[0],'bo')

plt.subplot(222)
plt.ylabel("R2 energy")
plt.xlabel("WE")
plt.plot(welist,r2list[1],'mo')

plt.subplot(223)
plt.ylabel("R2 energy")
plt.xlabel("R2 band")
plt.plot(r2list[0],r2list[1],'go')
plt.plot(r2list[0][6],r2list[1][6],'ks')
'''

#Plot bands and energies
dpp=dp( 0.90138347,0.32135407,0.50086193,0.04255208,0.62642983)
dpp[0]=np.array(dpp[0])
dpp[1]=np.array(dpp[1])
a,b=(0.01364334,0.01583295)

plt.suptitle("t,tt,K,J,m="+str([0.901,0.321,0.501,0.0426,0.626])+" eV")

plt.subplot(221)
plt.title("FLP bands")
plt.plot(d0[0][:90],'go',label="e_pbe0 + ef")
plt.plot(dpp[0][:90]+a,'bo',label="e_tb")
plt.legend(loc=2)

plt.subplot(222)
plt.title("COL bands")
plt.plot(d0[0][90:180],'go')
plt.plot(dpp[0][90:180]+a,'bo')

plt.subplot(223)
plt.title("FM bands")
plt.plot(d0[0][180:270],'go')
plt.plot(dpp[0][180:270]+a,'bo')

plt.subplot(224)
plt.title("Energies")
plt.ylabel("E_pred")
plt.xlabel("E_PBE0")
plt.plot(d0[1],d0[1],'g-')
plt.plot(d0[1],dpp[1]+b,'bo')

plt.show()
