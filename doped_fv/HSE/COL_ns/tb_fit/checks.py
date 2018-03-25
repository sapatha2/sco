import json
import linecache
'''
#Read and dump weights
w=[]
for r in range(111,137):
  w+=linecache.getline("../KRED.DAT",r).split()
  w=[float(z) for z in w]

f=open("weights.p","w")
json.dump(w,f)
print(sum(w))
print("Weight readin done")
'''
'''
#Read and dump eigenvalues and rho 
inp=open("../prop.in.o","r")
i=0
ef=0
e1=[]
e2=[]
rho1=[]
rho2=[]
kzz=[]
elow=[]

for line in inp:
  if i<2:
    if "EFERMI" in line:
      ef=float(line.split()[5])
    if "ALPHA" in line:
      i+=1
  else:
    if "BETA" in line: 
      break
    elif "EIGENVALUE" in line:
      #If kz=0
      evec=line.split()[6]
      if(int(evec[0])==0): kzz.append(1)
      else: kzz.append(0)
      #Append all eigenvalues from bands crossing fe
      for k in range(7): 
        line=inp.next()
      x=line.split()
      x=[float(y) for y in x]
      e1.append(x[6])
      #Occupation of states on these bands
      if(x[6]<=ef): rho1.append(1)
      else: rho1.append(0) 

i=0
inp=open("../prop.in.o","r")
for line in inp:
  if i<2:
    if "BETA" in line:
      i+=1
  else:
    if "EIGENVALUE" in line:
      evec=line.split()[6]
      #Append all eigenvalues from bands crossing fe
      for k in range(7): 
        line=inp.next()
      x=line.split()
      x=[float(y) for y in x]
      e2.append(x[6])
      #Occupation of states on these bands
      if(x[6]<=ef): rho2.append(1)
      else: rho2.append(0) 

inp=open("../prop.in.o","r")
#Read and dump low eigenvalues
for line in inp:
  if "EIGENVALUES - " in line:
    for k in range(7):
      line=inp.next()
      elow+=line.split()

e=[e1,e2]
rho=[rho1,rho2]
elow=[float(x) for x in elow]
f=open("evals.p","w")
g=open("rho.p","w")
h=open("elow.p","w")
i=open("kzz.p","w")
json.dump(e,f)
json.dump(rho,g)
json.dump(elow,h)
json.dump(kzz,i)

'''
import numpy as np 
import matplotlib.pyplot as plt

f=open("weights.p","r")
w=json.load(f)

f=open("evals.p","r")
e=json.load(f)

g=open("rho.p","r")
rho=json.load(g)

#h=open("elow.p","r")
#elow=json.load(h)

i=open("kzz.p","r")
kzz=json.load(i)

w=np.array(w)
rho=np.array(rho)

#Check 1: that the weights sum up to 1 
d={"Sum weights": sum(w),
#Check 2: that the sum of weights*density sums to 1 
  "Sum weights*rho": np.dot(w,rho[0]+rho[1]),
#Check 3: sum of weights*density where kz=0 only
  "Sum weights*rho, kz=0": np.dot(w*kzz,rho[0]+rho[1]),
#Check 4: histogram of low energy eigenvalues
#plt.hist(elow,bins=30)
#plt.show()
#Final calculation: 
  "Sum weights*rho*evals, kz=0": np.dot(w*kzz,rho[0]*e[0]+rho[1]*e[1]),
  "val": np.dot(w,rho[0]*e[0]+rho[1]*e[1])}

j=open("884.p","w")
json.dump(d,j)
