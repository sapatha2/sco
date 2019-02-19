#Generate input files for 2rdm, use same slater files as old 1rdm!
import os
import json 
import numpy as np 

def makejson(path):
  basename='b'+str(path)
  if(path==1): gsws=[1e-06,0.25,0.50,0.75,1.0,-0.25,-0.50,-0.75] #t/J sampling
  else: gsws=[0.25,0.50,0.75,1.0]

  for gsw in gsws:
    fname=basename+'/gsw'+str(gsw)+'.vmc'
    print(fname)
    
    #Make my JSON 
    print('command: ','~/mainline/bin/gosling '+fname+'.log -json &> '+fname+'.gosling.json')
    os.system('~/mainline/bin/gosling '+fname+'.log -json &> '+fname+'.gosling.json')
    
    #Replace labels in JSON 
    json_f=open(fname+'.gosling.json','r').read().split("\n")
    i=1
    for j in range(len(json_f)):
      if('tbdm_basis' in json_f[j]): 
        json_f[j]='"tbdm_basis'+str(i)+'":{'
        i+=1
    json_o=open(fname+'.gosling.json','w')
    json_o.write('\n'.join(json_f))

if __name__=='__main__':
  for path in np.arange(1,5): makejson(path)
