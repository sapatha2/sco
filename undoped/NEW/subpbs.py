import os 
import numpy as np 

def subpbs(path):
  basename='p'+str(path)
  for gsw in [0.0,0.25,0.50,0.75,1.0,-0.25,-0.50,-0.75]:  
    fname=basename+'/'+'gsw'+str(gsw)+'.vmc.pbs'
    os.system('qsub '+fname)

if __name__=='__main__':
  for path in [4,5,6]: subpbs(path)     
