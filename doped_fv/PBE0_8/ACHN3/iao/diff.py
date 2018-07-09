import cubetools
import copy 

for i in [10,20,30,40,50,60,70,80,810,820,830,840,850,860,870,880]:
  ud=cubetools.read_cube("ACHN_0.plot.orb"+str(i)+".cube")
  do=cubetools.read_cube("ACHN3_0.plot.orb"+str(i)+".cube")
  do['data']-=ud['data'] #make 'up' the spin density
  cubetools.write_cube(do,"DIFF_0.plot.orb"+str(i)+".cube")

