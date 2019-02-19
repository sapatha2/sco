#Generate input files for 2rdm, use same slater files as old 1rdm!
import os 
import shutil 
import numpy as np 

def geninput(path):
  '''
  input:
  path == 7 - FLP to FLP1
  path == 8 - FLP to FLP2
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
  if(path==7): gsws=[1e-6,0.25,0.50,0.75,1.0]
  else:        gsws=[0.25,0.50,0.75,1.0]
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
    '#PBS -l walltime=04:00:00\n'+\
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
    'include all_flp.sys\n'+\
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

    #FLP
    flpu=np.arange(1,68)
    flpd=np.arange(69,69+65)
    flp='  '+' '.join([str(x) for x in flpu])
    flp+='\n'
    flp+='  '+' '.join([str(x) for x in flpd])
    flp+='\n\n'    

    #FLP1
    flp1u=np.arange(1,68)
    flp1d=list(np.arange(69,69+64))+[134]
    flp1='  '+' '.join([str(x) for x in flp1u])
    flp1+='\n'
    flp1+='  '+' '.join([str(x) for x in flp1d])
    flp1+='\n\n'    

    #FLP2
    flp2u=np.arange(1,68)
    flp2d=list(np.arange(69,69+54))+list(np.arange(69+55,69+65))+[134]
    flp2='  '+' '.join([str(x) for x in flp2u])
    flp2+='\n'
    flp2+='  '+' '.join([str(x) for x in flp2d])
    flp2+='\n\n'    

    if("7" in basename): states=flp+flp1
    elif("8" in basename): states=flp2+flp
    else:
      print("Dont have this path: ", basename)
      exit(0)

    #Make input file
    string='SLATER\n'+\
    'ORBITALS  {\n'+\
    '  MAGNIFY 1.0\n'+\
    '  NMO 136\n'+\
    '  ORBFILE all_flp.orb\n'+\
    '  INCLUDE all_flp.basis\n'+\
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
  for path in [7,8]:
    geninput(path)
