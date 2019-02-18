#Generate input files for 2rdm, use same slater files as old 1rdm!
import os 
import shutil 
import numpy as np 

def geninput(path):
  '''
  input:
  path == 4 - FLP to FLPp
  path == 5 - FLPp to FLPpp
  path == 6 - FLPpp to FLP  
  path == 17 - FLPup to FLP
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

def genslater(basename):
  for gsw in [0.25,0.50,0.75,1.0]:
    fname='gsw'+str(gsw)
    w=[np.sign(gsw)*np.sqrt(abs(gsw)),np.sqrt(1-abs(gsw))]

    #FLP
    flpu=np.arange(1,68)
    flpd=np.arange(69,69+65)
    flp='  '+' '.join([str(x) for x in flpu])
    flp+='\n'
    flp+='  '+' '.join([str(x) for x in flpd])
    flp+='\n\n'    

    #FLPp
    flppu=np.arange(1,68)
    flppd=list(np.arange(69,69+64))+[134]
    flpp='  '+' '.join([str(x) for x in flppu])
    flpp+='\n'
    flpp+='  '+' '.join([str(x) for x in flppd])
    flpp+='\n\n'    

    #FLPpp
    flpppu=np.arange(1,68)
    flpppd=list(np.arange(69,69+64))+[135]
    flppp='  '+' '.join([str(x) for x in flpppu])
    flppp+='\n'
    flppp+='  '+' '.join([str(x) for x in flpppd])
    flppp+='\n\n'    

    #FLPpLO
    flpplou=np.arange(1,68)
    flpplod=list(np.arange(69,69+62))+[132,133,134]
    flpplo='  '+' '.join([str(x) for x in flpplou])
    flpplo+='\n'
    flpplo+='  '+' '.join([str(x) for x in flpplod])
    flpplo+='\n\n'    

    #FLPppLO
    flppplou=np.arange(1,68)
    flppplod=list(np.arange(69,69+62))+[132,133,136]
    flppplo='  '+' '.join([str(x) for x in flppplou])
    flppplo+='\n'
    flppplo+='  '+' '.join([str(x) for x in flppplod])
    flppplo+='\n\n'    

    #FLPup
    flpupu=list(np.arange(1,67))+[68]
    flpupd=np.arange(69,69+65)
    flpup='  '+' '.join([str(x) for x in flpupu])
    flpup+='\n'
    flpup+='  '+' '.join([str(x) for x in flpupd])
    flpup+='\n\n'    

    if("4" in basename): states=flp+flpp
    elif("5" in basename): states=flpp+flppp
    elif("6" in basename): states=flppp+flp
    elif("11" in basename): states=flpplo+flpp
    elif("12" in basename): states=flppplo+flppp
    elif("17" in basename): states=flpup+flp
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
  #for path in np.arange(4,7): 
  for path in [17]:
    geninput(path)
