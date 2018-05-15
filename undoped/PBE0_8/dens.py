from crystal2pyscf import crystal2pyscf_mol, crystal2pyscf_cell
from pyscf.scf.hf import make_rdm1
from pyscf.scf.hf import get_ovlp
from pyscf import gto, scf, lo
import numpy as np 
from pyscf.lo import orth
from functools import reduce
import scipy as sp 
import matplotlib.pyplot as plt

###########################################################################################
#Basis

basis={'Cu':gto.basis.parse('''
  Cu S
  27.8467870 -0.019337
  21.4206050 0.088614
  10.5372500 -0.355859
  2.53559600 0.538088
  1.24968400 0.511137
  0.59984600 0.147640
  Cu S
  27.8467870 -0.002653
  21.4206050 -0.007542
  10.5372500 0.079756
  2.53559600 -0.155301
  1.24968400 -0.253078
  0.59984600 -0.028168
  Cu P
  18.5940060 0.003652
  14.3030110 -0.037076
  5.63142000 0.217120
  2.68301100 0.440973
  1.25857300 0.358723
  0.57408400 0.114846
  0.20923000 0.007316
  Cu D
  15.7872870 0.084222
  7.20321300 0.255433
  2.97360600 0.349966
  1.20702400 0.341672
  0.46404600 0.241697
  Cu F
  1.6923298836 1.00000
  Cu S
  0.2 1.0
  Cu S
  0.6 1.0
  Cu P 
  0.2 1.0
  Cu P 
  0.6 1.0
  Cu D 
  0.2 1.0
  Cu D
  0.6 1.0'''),
'O':gto.basis.parse('''
  O S
  0.268022 0.304848
  0.573098 0.453752
  1.225429 0.295926
  2.620277 0.019567
  5.602818 -0.128627
  11.980245 0.012024
  25.616801 0.000407
  54.775216 -0.000076
  O P
  0.333673 0.255999
  0.666627 0.281879
  1.331816 0.242835
  2.660761 0.161134
  5.315785 0.082308
  10.620108 0.039899
  21.217318 0.004679
  O D
  0.669340 1.000000
  O F
  1.423104 1.000000
  O S
  0.2 1.0
  O S
  0.6 1.0
  O P 
  0.2 1.0
  O P 
  0.6 1.0
'''),
'Sr':gto.basis.parse('''
  Sr S
  0.217685 -0.093894
  0.487102 -0.247613
  1.089961 0.160101
  2.438947 -0.043595
  5.457499 0.008684
  12.211949 -0.001296
  Sr P 
  0.223593 -0.047473
  0.450721 -0.171028
  0.908566 0.104848
  1.831493 -0.029516
  3.691937 0.006196
  7.442231 -0.000955
  Sr D
  0.528269 1.000000
  Sr S
  0.2 1.0
  Sr S
  0.6 1.0
  Sr P
  0.2 1.0
  Sr P 
  0.6 1.0
  ''')}
basis_order = {'Cu':[0,0,1,2,3,0,0,1,1,2,2],'O':[0,1,2,3,0,0,1,1],'Sr':[0,1,2,0,0,1,1]}

###########################################################################################
#Methods

def compMolCell():
  '''
  Builds from CRYSTAL calculation a MOL and CELL object, and respective SCF objects.
  Prints the following quantities for both MOL and CELL
  1. Trace of MO orbital overlap matrix - should be N_bas
  2. Trace of 1rdm on MO basis - should be N_elec in each spin sector
  
  In addition there is a print of the difference between the two objects:
  1. sum(abs(s_cell - s_mol)), where s is the overlap matrix
  2. sum(abs(mo_cell - mo_mol)), where mo is the MO coefficient matrix
  '''
  
  ########################################################
  #MOL readin - INCORRECT IF YOU HAVE A PBC CYRSTAL RUN
  mol,mf=crystal2pyscf_mol(basis=basis,basis_order=basis_order)

  s=mf.get_ovlp()
  mo_up=mf.mo_coeff[0]
  mo_dn=mf.mo_coeff[1]

  #MO OVERLAPS
  mo_ovlpu=reduce(np.dot,(mo_up.T,s,mo_up))
  mo_ovlpd=reduce(np.dot,(mo_dn.T,s,mo_dn))
  print("MOL: Tr(MO_ovlp)",np.trace(mo_ovlpu),np.trace(mo_ovlpd))

  #1RDM ON MO
  rdm1=mf.make_rdm1(mf.mo_coeff,mf.mo_occ)
  mo_dmu=reduce(np.dot,(mo_up.T,s,rdm1[0],s,mo_up))
  mo_dmd=reduce(np.dot,(mo_dn.T,s,rdm1[1],s,mo_dn))
  print("MOL: Tr(MO_dens)",np.trace(mo_dmu),np.trace(mo_dmd))

  ########################################################
  #CELL readin - CORRECT IF YOU HAVE A PBC CRYSTAL RUN
  mol,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order)
  print("sum(abs(s_cell-s_mol))",np.sum(np.abs(mf.get_ovlp()[0]-s)))
  print("sum(abs(mo_cell-mo_mol))",np.sum(np.abs(mf.mo_coeff[0][0]-mo_up)),np.sum(np.abs(mf.mo_coeff[1][0]-mo_dn)))

  s=mf.get_ovlp()[0]
  mo_up=mf.mo_coeff[0][0]
  mo_dn=mf.mo_coeff[1][0]

  #MO OVERLAPS
  mo_ovlpu=reduce(np.dot,(mo_up.T,s,mo_up))
  mo_ovlpd=reduce(np.dot,(mo_dn.T,s,mo_dn))
  print("CELL: Tr(MO_ovlp)",np.trace(mo_ovlpu),np.trace(mo_ovlpd))
  #print((mo_ovlpu-np.identity(800)).max())

  #1RDM ON MO
  rdm1=mf.make_rdm1(mf.mo_coeff,mf.mo_occ)
  mo_dmu=reduce(np.dot,(mo_up.T,s,rdm1[0][0],s,mo_up))
  mo_dmd=reduce(np.dot,(mo_dn.T,s,rdm1[1][0],s,mo_dn))
  print("CELL: Tr(MO_dens)",np.trace(mo_dmu),np.trace(mo_dmd))
  #print((mo_dmu[:132,:132]-np.identity(132)).max())
  #print((mo_dmd[:132,:132]-np.identity(132)).max())


def calcOAO(mol,mf):
  '''
  input: 
    mol and scf (PBC SCF) objects from pyscf 
  output:
    Calculates 1RDM on orthog atomic orbitals, orthogonalized
    using Lowdin S^1/2
    Returns (rdm1up+rd1mdn, rdm1up-rdm1dn)
  '''

  s=mf.get_ovlp()[0]
  s12=sp.linalg.sqrtm(s).real 
  rdm1=mf.make_rdm1(mf.mo_coeff,mf.mo_occ)
  oao_dmu=reduce(np.dot,(s12,rdm1[0][0],s12))
  oao_dmd=reduce(np.dot,(s12,rdm1[1][0],s12))
  oao_dmu+=oao_dmd
  oao_dmd=oao_dmu-2*oao_dmd
  
  #Printouts. nup+ndown and nup-ndown
  '''
  print("CELL: Tr(OAO_dens) - Tot ",np.trace(oao_dmu),np.trace(oao_dmd))
  print("CELL: Tr(OAO_dens) - Cu 1",np.trace(oao_dmu[:35,:35]),np.trace(oao_dmd[:35,:35]))
  print("CELL: Tr(OAO_dens) - Cu 2",np.trace(oao_dmu[35:70,35:70]),np.trace(oao_dmd[35:70,35:70]))
  print("CELL: Tr(OAO_dens) - O 1 ",np.trace(oao_dmu[280:304,280:304]),np.trace(oao_dmd[280:304,280:304]))
  print("CELL: Tr(OAO_dens) - O 2 ",np.trace(oao_dmu[304:328,304:328]),np.trace(oao_dmd[304:328,304:328]))
  print("CELL: Tr(OAO_dens) - Sr 1",np.trace(oao_dmu[664:681,664:681]),np.trace(oao_dmd[664:681,664:681]))
  '''
  return (oao_dmu,oao_dmd)

def subOAO(oao_dmu,oao_dmd):
  '''
  input: up and down density matrices on OAO basis
  output: sub-density matrices on different types of atoms
  '''
  ncu=35 #Number of basis elements per copper
  no=24  #Number of basis elements per oxygen
  nsr=17 #Number of basis elements per strontium
  
  '''
  cu_oao=[]
  #Construct the sub DM for copper
  for i in range(8):
    cu_oao.append([oao_dmu[i*ncu:(i+1)*ncu,i*ncu:(i+1)*ncu],oao_dmd[i*ncu:(i+1)*ncu,i*ncu:(i+1)*ncu]])

  #Plot sub DM for copper
  plt.suptitle("Cu OAO occupations")
  plt.subplot(121)
  plt.title("Nup+Ndown") 
  plt.xticks(np.arange(0,ncu,1),mol.sph_labels()[:ncu],rotation='vertical')
  for i in range(8):
    plt.plot(np.diag(cu_oao[i][0]),'o')
  plt.subplot(122)
  plt.title("Nup-Ndown") 
  plt.xticks(np.arange(0,ncu,1),mol.sph_labels()[:ncu],rotation='vertical')
  for i in range(8):
    plt.plot(np.diag(cu_oao[i][1]),'o')
  plt.show()
  '''
  
  o_oao=[]
  #Construct the sub DM for copper
  for i in range(16):
    o_oao.append([oao_dmu[280+i*no:280+(i+1)*no,280+i*no:280+(i+1)*no],oao_dmd[280+i*no:280+(i+1)*no,280+i*no:280+(i+1)*no]])

  #Plot sub DM for copper
  plt.suptitle("O OAO occupations")
  plt.subplot(121)
  plt.title("Nup+Ndown") 
  plt.xticks(np.arange(0,no,1),mol.sph_labels()[280:280+no],rotation='vertical')
  for i in range(16):
    plt.plot(np.diag(o_oao[i][0]),'o')
  plt.subplot(122)
  plt.title("Nup-Ndown") 
  plt.xticks(np.arange(0,no,1),mol.sph_labels()[280:280+no],rotation='vertical')
  for i in range(16):
    plt.plot(np.diag(o_oao[i][1]),'o')
  plt.show()

###########################################################################################
#Run
direc="COL"

#compMolCell()
mol,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=direc+"/GRED.DAT",kred=direc+"/KRED.DAT",cryoutfn=direc+"/prop.in.o")
oao_dmu,oao_dmd=calcOAO(mol,mf)
subOAO(oao_dmu,oao_dmd)
