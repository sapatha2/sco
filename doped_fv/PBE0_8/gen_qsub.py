import subprocess
import os 

for root in ["CHK0","CHK2","COL0","COL2",'FLP0','FLP2','FLP4','BCOL0','BCOL2','BLK0','BLK2','ACHN2','ACHN4','ACHN6','FM6','FM8','FM10']:
  #Write qsub files
  '''
  f=open(root+"/crys.qsub","w")
  f.write("#PBS -q wagner \n"+\
 "#PBS -l nodes=1,flags=allprocs \n"+\
 "#PBS -l walltime=24:00:00 \n"+\
 "#PBS -N sco_"+root+"_p8 \n"+\
 "#PBS -j oe \n"+\
 "#PBS -o sco_"+root+"_p8.jobout  \n"+\
 "#PBS -e sco_"+root+"_p8.err \n"+\
 
 "module load openmpi/1.4-gcc+ifort \n"+\
 "module load intel/14.0 \n"+\
 "cd ${PBS_O_WORKDIR} \n"+\
 "mpiexec ../../../../CRYSTAL17/bin/Linux-ifort14_XE_emt64/v1.0.1/Pcrystal INPUT &> OUTPUT \n")
  f.close()
  '''
  #Copy to INPUT
  #subprocess.Popen(['cp',root+'/crys_pbe0.in',root+'/INPUT'])
  
  #Submit
  os.chdir("./"+root)
  subprocess.Popen(['qsub','crys.qsub'])
  os.chdir("../")

print("Done")
