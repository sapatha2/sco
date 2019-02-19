#Generate input files for 2rdm, use same slater files as old 1rdm!
import os 
import shutil 
import numpy as np 

def geninput(path):
  '''
  input:
  path == 1 - CHK to COL
  path == 2 - CHK1 to CHK
  path == 3 - CHK2 to CHK  
  path == 4 - CHK3 to CHK
  output:
  slater files
  vmc files 
  pbs files for vmc
  '''
 
  basename='b'+str(path)
  #Make sure directory exists, else make it 
  if not os.path.exists(basename):
    #Move all important files to there 
    shutil.copytree('req_files',basename+'/') 
  else:
    print('Directory '+str(basename)+' exists, not going to overwrite')
    exit(0)  
  
  if(path==1): gsws=[1e-6,0.25,0.50,0.75,1.0,-0.25,-0.50,-0.75] #t/J sampling
  else: gsws=[0.25,0.50,0.75,1.0] 
  genslater(basename,gsws)
  genvmc(basename,gsws)
  genpbs(basename,gsws)
  return 

def genpbs(basename,gsws):
  for gsw in gsws:
    fname='gsw'+str(gsw)
   
    #Blue waters input  
    string='#!/bin/bash\n'+\
    '#PBS -q low\n'+\
    '#PBS -l nodes=8:ppn=32:xe\n'+\
    '#PBS -l walltime=10:00:00\n'+\
    '#PBS -N '+basename+'/'+fname+'.vmc\n'\
    '#PBS -e '+basename+'/'+fname+'.vmc.perr\n'+\
    '#PBS -o '+basename+'/'+fname+'.vmc.pout\n'+\
    'mkdir -p /scratch/sciteam/$USER/sco/undoped/NEW/'+basename+'\n'+\
    'cd /scratch/sciteam/$USER/sco/undoped/NEW/'+basename+'\n'+\
    'cp -u /u/sciteam/$USER/sco/undoped/NEW/'+basename+'/* .'+'\n'+\
    'aprun -n 256 /u/sciteam/$USER/fork/bin/qwalk '+fname+'.vmc &> '+fname+'.vmc.out\n'

    f=open(basename+'/'+fname+'.vmc.pbs','w')
    f.write(string)
    f.close()      
  return 1

def genvmc(basename,gsws):
  for gsw in gsws:
    fname='gsw'+str(gsw)
    
    string='method {\n'+\
    '  vmc\n'+\
    '  nblock 250\n'+\
    '  nstep 25\n'+\
    '  average { tbdm_basis\n'+\
    '    mode tbdm_diagonal\n'+\
    '    orbitals {\n'+\
    '      magnify 1\n'+\
    '      nmo 76\n'+\
    '      orbfile iao_0.orb\n'+\
    '      include iao.basis\n'+\
    '      centers { useglobal }\n'+\
    '    }\n'+\
    '    states { 14 24 34 44 }\n'+\
    '  }\n'+\
    '  average { tbdm_basis\n'+\
    '    mode obdm\n'+\
    '    orbitals {\n'+\
    '      magnify 1\n'+\
    '      nmo 76\n'+\
    '      orbfile iao_0.orb\n'+\
    '      include iao.basis\n'+\
    '      centers { useglobal }\n'+\
    '    }\n'+\
    '    states { 14 24 34 44 46 50 54 58 63 67 71 75 5 15 25 35 }\n'+\
    '  }\n'+\
    '  average { tbdm_basis\n'+\
    '    mode obdm\n'+\
    '    orbitals {\n'+\
    '      magnify 1\n'+\
    '      nmo 76\n'+\
    '      orbfile iao_0.orb\n'+\
    '      include iao.basis\n'+\
    '      centers { useglobal }\n'+\
    '    }\n'+\
    '    states { 47 51 55 59 62 66 70 74 }\n'+\
    '  }\n'+\
    '}\n'+\
    '\n'+\
    'include sys\n'+\
    'trialfunc {\n'+\
    '  slater-jastrow\n'+\
    '  wf1 { include '+fname+'.slater }\n'+\
    '  wf2 { include optjast3 }\n'+\
    '}\n'

    f=open(basename+'/'+fname+'.vmc','w')
    f.write(string)
    f.close()      
  return 1

def genslater(basename,gsws):
  for gsw in gsws:  
    fname='gsw'+str(gsw)
    w=[np.sign(gsw)*np.sqrt(abs(gsw)),np.sqrt(1-abs(gsw))]
 
    #CHK
    chku=np.arange(1,67)
    chkd=np.arange(135,135+66)
    chk='  '+' '.join([str(x) for x in chku])
    chk+='\n'
    chk+='  '+' '.join([str(x) for x in chkd])
    chk+='\n\n'    

    #COL
    colu=np.arange(68,68+66)
    cold=np.arange(202,202+66)
    col='  '+' '.join([str(x) for x in colu])
    col+='\n'
    col+='  '+' '.join([str(x) for x in cold])
    col+='\n\n'    
    
    #CHK1
    chk1u=list(np.arange(1,66))+[67]
    chk1d=np.arange(135,135+66)
    chk1='  '+' '.join([str(x) for x in chk1u])
    chk1+='\n'
    chk1+='  '+' '.join([str(x) for x in chk1d])
    chk1+='\n\n'    

    #CHK2
    chk2u=list(np.arange(1,52))+list(np.arange(53,67))+[67]
    chk2d=np.arange(135,135+66)
    chk2='  '+' '.join([str(x) for x in chk2u])
    chk2+='\n'
    chk2+='  '+' '.join([str(x) for x in chk2d])
    chk2+='\n\n'    
    
    #CHK3
    chk3u=list(np.arange(1,48))+list(np.arange(49,67))+[67]
    chk3d=np.arange(135,135+66)
    chk3='  '+' '.join([str(x) for x in chk3u])
    chk3+='\n'
    chk3+='  '+' '.join([str(x) for x in chk3d])
    chk3+='\n\n'    

    if("1" in basename): states=chk+col
    elif("2" in basename): states=chk1+chk
    elif("3" in basename): states=chk2+chk
    elif("4" in basename): states=chk3+chk
    else:
      print("Dont have this path: ", basename)
      exit(0)

    #Make input file
    string='SLATER\n'+\
    'ORBITALS  {\n'+\
    '  MAGNIFY 1.0\n'+\
    '  NMO 268\n'+\
    '  ORBFILE all_0.orb\n'+\
    '  INCLUDE all.basis\n'+\
    '  CENTERS { USEGLOBAL }\n'+\
    '}\n'+\
    '\n'+\
    'SHERMAN_MORRISON_UPDATES \n'+\
    'DETWT { \n' + '\n'.join(['  '+str(x) for x in w])+' \n}\n'+\
    'STATES {\n'+\
    ''.join(states)+\
    '}\n'
    f=open(basename+'/'+fname+'.slater','w')
    f.write(string)
    f.close()
  return 1

if __name__=='__main__':
  for path in [1,2,3,4]:
    geninput(path)
