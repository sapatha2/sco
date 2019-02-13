#Generate input files for 2rdm, use same slater files as old 1rdm!
import os 
import shutil 
import numpy as np 

def geninput(path):
  '''
  input:
  path == 7 - FM to FMp
  path == 8 - FMp to FMp2
  path == 9 - FMp2 to FM
  output:
  slater files
  vmc files 
  pbs files for vmc
  '''
 
  basename='p'+str(path)
  #Make sure directory exists, else make it 
  if not os.path.exists(basename):
    #Move all important files to there 
    shutil.copytree('req_files',basename+'/') 
  else:
    print('Directory '+str(basename)+' exists, not going to overwrite')
    exit(0)  
  genslater(basename)
  genvmc(basename)
  genpbs(basename)
  return 

def genpbs(basename):
  for gsw in [0.25,0.50,0.75,1.0]:
    fname='gsw'+str(gsw)
   
    #Blue waters input  
    string='#!/bin/bash\n'+\
    '#PBS -q low\n'+\
    '#PBS -l nodes=8:ppn=32:xe\n'+\
    '#PBS -l walltime=08:00:00\n'+\
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

def genvmc(basename):
  for gsw in [0.25,0.50,0.75,1.0]:
    fname='gsw'+str(gsw)
    
    string='method {\n'+\
    '  vmc\n'+\
    '  nblock 250\n'+\
    '  nstep 25\n'+\
    '  average { tbdm_basis\n'+\
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
    '    states { 14 24 34 44 46 50 54 58 63 67 71 75 }\n'+\
    '  }\n'+\
    '}\n'+\
    '\n'+\
    'include all_fm.sys\n'+\
    'trialfunc {\n'+\
    '  slater-jastrow\n'+\
    '  wf1 { include '+fname+'.slater }\n'+\
    '  wf2 { include optjast3 }\n'+\
    '}\n'

    f=open(basename+'/'+fname+'.vmc','w')
    f.write(string)
    f.close()      
  return 1

def genslater(basename):
  for gsw in [0.25,0.50,0.75,1.0]:
    fname='gsw'+str(gsw)
    w=[np.sign(gsw)*np.sqrt(abs(gsw)),np.sqrt(1-abs(gsw))]

    #FM
    fmu=np.arange(1,69)
    fmd=np.arange(69,69+64)
    fm='  '+' '.join([str(x) for x in fmu])
    fm+='\n'
    fm+='  '+' '.join([str(x) for x in fmd])
    fm+='\n\n'    

    #FMp
    fmpu=np.arange(1,69)
    fmpd=list(np.arange(69,69+63))+[69+64]
    fmp='  '+' '.join([str(x) for x in fmpu])
    fmp+='\n'
    fmp+='  '+' '.join([str(x) for x in fmpd])
    fmp+='\n\n'    

    #FMp2
    fmppu=np.arange(1,69)
    fmppd=list(np.arange(69,69+63))+[69+65]
    fmpp='  '+' '.join([str(x) for x in fmppu])
    fmpp+='\n'
    fmpp+='  '+' '.join([str(x) for x in fmppd])
    fmpp+='\n\n'    

    if("7" in basename): states=fm+fmp
    elif("8" in basename): states=fmp+fmpp
    elif("9" in basename): states=fmpp+fm
    else:
      print("Dont have this path: ", basename)
      exit(0)

    #Make input file
    string='SLATER\n'+\
    'ORBITALS  {\n'+\
    '  MAGNIFY 1.0\n'+\
    '  NMO 136\n'+\
    '  ORBFILE all_fm.orb\n'+\
    '  INCLUDE all_fm.basis\n'+\
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
  for path in np.arange(7,10): 
    geninput(path)
