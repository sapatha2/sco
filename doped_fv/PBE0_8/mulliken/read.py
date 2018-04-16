import subprocess
import numpy as np 
import json 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
'''
#Read in mulliken population from files
#UNDOPED
#for root in ["CHK","COL","FLP","BCOL","BLK","ACHN","FM"]:
#  path="../../../undoped/PBE0_8/"+root+"/OUTPUT"
#  s=subprocess.Popen(["grep","TOTAL ENERGY",path],stdout=subprocess.PIPE).communicate()[0]
#  n=int(s.decode("utf-8").split("\n")[1].split()[2][:2])

#DOPED
for root in ["COL0","FLP2","ACHN2","CHK2","BLK0","FLP0","BCOL2","CHK0","FLP4","ACHN6","BCOL0","ACHN4","FM8","FM10"]:
  path="../"+root+"/OUTPUT"
  s=subprocess.Popen(["grep","TOTAL ENERGY",path],stdout=subprocess.PIPE).communicate()[0]
  n=int(s.decode("utf-8").split("\n")[2].split()[2][:2])

  i=0
  f=open(path,"r")
  cline=[]
  sline=[]
  for line in f:
    if "TOTAL ATOMIC CHARGES" in line:
      i+=1
      if(i==n+1):
        #Pass one
        line=next(f)
        cline+=line.split()
        line=next(f)
        cline+=line.split()[:2]
        
        for j in range(7):
          line=next(f)
        sline+=line.split()
        line=next(f)
        sline+=line.split()[:2]

    else:
      pass

  #Write to matrices
  cline=[float(x) for x in cline]
  sline=[float(x) for x in sline]

  c=[[0,0,cline[0],cline[4],0,0],
     [0,cline[5],cline[3],cline[7],cline[1],0],
     [cline[0],cline[4],cline[2],cline[6],cline[0],cline[4]],
     [0,cline[7],cline[1],cline[5],cline[3],0],
     [0,0,cline[0],cline[4],0,0]]
  
  s=[[0,0,sline[0],sline[4],0,0],
     [0,sline[5],sline[3],sline[7],sline[1],0],
     [sline[0],sline[4],sline[2],sline[6],sline[0],sline[4]],
     [0,sline[7],sline[1],sline[5],sline[3],0],
     [0,0,sline[0],sline[4],0,0]]
  
  pop=[c,s] #pop[0]=c, pop[1]=s

  jf=open(root+".json","w")
  json.dump(pop,jf)

'''
#Read and plot undoped
#for root in ["CHK","COL","FLP","BCOL","BLK","ACHN","FM"]:
#for root in ["COL0","FLP2","ACHN2","CHK2","BLK0","FLP0","BCOL2","CHK0","FLP4","ACHN6","BCOL0","ACHN4","FM8","FM10"]:
for roots in [["COL0","COL"],['FLP2','FLP'],['ACHN2','ACHN'],['CHK2','CHK'],['BLK0','BLK'],['FLP0','FLP'],['BCOL2','BCOL'],
  ['CHK0','CHK'],['FLP4','FLP'],['ACHN6','ACHN'],['BCOL0','BCOL'],['ACHN4','ACHN'],['FM8','FM'],['FM10','FM']]:

  popd=json.load(open(roots[0]+".json","r"))
  popu=json.load(open(roots[1]+".json","r"))

  fig=plt.figure()
  axc=fig.add_subplot(2,1,1)
  plt.matshow(np.array(popd[0])-np.array(popu[0]),fignum=False)
  plt.grid()
  plt.colorbar()
  axc.add_patch(patches.Polygon([[4.2,2], [2.5,0.3], [0.8,2],[2.5,3.7]],fill=False))
  
  axs=fig.add_subplot(2,1,2)
  plt.matshow(np.array(popd[1])-np.array(popu[1]),fignum=False)
  plt.grid()
  plt.colorbar()
  axs.add_patch(patches.Polygon([[4.2,2], [2.5,0.3], [0.8,2],[2.5,3.7]],fill=False))
  
  plt.suptitle("Mulliken Charge and Spin Diff: "+roots[0]+"-"+roots[1])
  fig.savefig(roots[0]+"_d.pdf",bbox_inches='tight')
