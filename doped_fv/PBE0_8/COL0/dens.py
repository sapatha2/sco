import numpy as np 
import json
'''
#Get density matrix, save to dens.p
apb=[] #ALPHA plus BETA
amb=[] #ALPHA minus BETA
swit=0
with open("xml") as f:
  for line in f:
    if("DIRECT_DENSITY_MATRIX__IVDL.1 " in line):
      if(not swit):
        for i in range(160000):
          x=next(f).split()
          x=[float(y) for y in x]
          apb+=x
        swit=1
        print("Finished A+B")
      else:
        for i in range(160000):
          x=next(f).split()
          x=[float(y) for y in x]
          amb+=x
        print("Finished A-B")
        break

mats=[apb,amb]
json.dump(mats,open("dens.p","w"))
'''

'''
#Get overlap matrix, save to overlap.p
S=[] #overlap 
with open("xml") as f:
  for line in f:
    if("DIRECT_OVERLAP_MATRIX__IVDL.1 " in line):
      for i in range(160000):
        x=next(f).split()
        x=[float(y) for y in x]
        S+=x
      print("Finished S")
      break

json.dump(S,open("ovlp.p","w"))
'''

#Diagonals
nelec=0
spin=0

mats=json.load(open("dens.p","r"))
apb,amb=mats
apb=np.reshape(apb,(-1,800))
amb=np.reshape(amb,(-1,800))
a=(apb+amb)/2.
b=(apb-amb)/2.

print(sum(np.diag(a)))
print(sum(np.diag(b)))

mats0=json.load(open("../../../undoped/PBE0_8/COL/dens.p","r"))
apb0,amb0=mats0
apb0=np.reshape(apb0,(-1,800))
amb0=np.reshape(amb0,(-1,800))
a0=(apb0+amb0)/2.
b0=(apb0-amb0)/2.

print(sum(np.diag(a0)))
print(sum(np.diag(b0)))

