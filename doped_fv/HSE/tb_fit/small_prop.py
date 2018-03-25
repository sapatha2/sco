import linecache

#Extract only the important parts of prop.in.o
i=0
j=0
e1=[]
e2=[]
w1=[]
w2=[]

#ef=-4.5395275E-02 #HSE, FLP
#ef=-5.1099662E-02 #HSE, FM
ef=-5.3308992E-02 #HSE, COL

w=[]
#for r in range(255,353):
#for r in range(318,447):
for r in range(255,353):
  w+=linecache.getline("../COL_ns/KRED.DAT",r).split()
w=[float(z) for z in w]
print("Weight readin done")

#Eigenvalues
el=0.
f=open("../COL_ns/prop.in.o","r")
for line in f:
  if(i<2):
    if "BETA" in line:
      i+=1
  else:
    if "EIGENVALUES" in line:
      #Only append if in kx-ky plane
      evec=line.split()[6]
      if(int(evec[0])==0):
        for k in range(7):
          line=f.next()
        x=line.split()
        x=[float(y) for y in x]
       
        if(x[6]<=ef):
          el+=x[6]*w[j]
        #if(x[5]<=ef):
        #  el+=x[5]*w[j]
      j+=1

print("BETA readin done")

i=0
j=0
f=open("../COL_ns/prop.in.o","r")
for line in f:
  if(i<2):
    if "ALPHA" in line:
      i+=1
  else:
    if "EIGENVALUES" in line:
      #Only append if in kx-ky plane
      evec=line.split()[6]
      if(int(evec[0])==0):
        for k in range(7):
          line=f.next()
        x=line.split()
        x=[float(y) for y in x]
        
        if(x[6]<=ef):
          el+=x[6]*w[j]
      j+=1
    if "BETA" in line: 
      break
print("ALPHA readin done")
print(el)
