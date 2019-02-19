import os 
import numpy as np 

def subpbs(path):
  basename='b'+str(path)
  if(path==1): gsws=[1e-6,0.25,0.50,0.75,1.0,-0.25,-0.50,-0.75] #t/J sampling
  else: gsws=[0.25,0.50,0.75,1.0]  
  for gsw in gsws:  
    fname=basename+'/'+'gsw'+str(gsw)+'.vmc.pbs'
    os.system('qsub '+fname)

if __name__=='__main__':
  for path in [1,2,3,4]: subpbs(path)
