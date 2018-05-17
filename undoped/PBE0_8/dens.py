from crystal2pyscf import crystal2pyscf_mol, crystal2pyscf_cell
from pyscf.scf.hf import make_rdm1
from pyscf.scf.hf import get_ovlp
from pyscf import gto, scf, lo
import numpy as np 
from pyscf.lo import orth
from functools import reduce
import scipy as sp 
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_pbc
from pyscf.lo.iao import iao

###########################################################################################
#Basis

#CUTOFF BFD basis
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

minbasis={'Cu':gto.basis.parse('''
  Cu S
  27.8467870 -0.019337
  21.4206050 0.088614
  10.5372500 -0.355859
  2.53559600 0.538088
  1.24968400 0.511137
  0.59984600 0.147640
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
  '''),
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
  '''),
'Sr':gto.basis.parse('''
  Sr S
  0.217685 -0.093894
  0.487102 -0.247613
  1.089961 0.160101
  2.438947 -0.043595
  5.457499 0.008684
  12.211949 -0.001296
  ''')}


#0.2 BFD basis
basis2={
'Cu':gto.basis.parse('''
  Cu S
  0.4 -0.0364413530365
  0.8 0.358763832032
  1.6 0.474343247928
  3.2 0.498447510592
  6.4 -0.270094109886
  12.8 -0.19207249392
  25.6 0.0629633270568
  51.2 -0.00853551842135
  102.4 0.000812242437275
  Cu S
  0.4 -2.28170595665
  0.8 1.46285008448
  1.6 -1.09989311533
  3.2 0.296788655986
  6.4 -0.12153417324
  12.8 0.117898710111
  25.6 -0.0341363112541
  51.2 0.00595066976203
  102.4 -0.00071519088395
  Cu P
  18.594006 0.003652
  14.303011 -0.037076
  5.63142 0.21712
  2.683011 0.440973
  1.258573 0.358723
  0.574084 0.114846
  0.20923 0.007316
  Cu D
  0.4 -0.117221524191
  0.8 0.420466340314
  1.6 0.166556512124
  3.2 0.31519384244
  6.4 0.201127125456
  12.8 0.128021225822
  25.6 0.00894474569825
  51.2 -0.00051734713442
  102.4 2.47330807683e-05
  Cu F
  1.6923298836 1.00000
  Cu S
  0.2 1.0
  Cu S
  0.6000000000000001 1.0
  Cu P
  0.2 1.0
  Cu P
  0.6000000000000001 1.0
  Cu D
  0.2 1.0
  Cu D
  0.6000000000000001 1.0
  '''),
'O':gto.basis.parse('''
  O S
  0.4 0.351884869096
  0.8 0.395141155066
  1.6 0.179458356356
  3.2 -0.0432052834993
  6.4 -0.105221770065
  12.8 0.0197823639914
  25.6 -0.00197386728468
  51.2 0.000387842127399
  102.4 -5.57569955803e-05
  O P 
  0.4 0.205849797904
  0.8 0.303195415412
  1.6 0.218836382326
  3.2 0.136227390096
  6.4 0.0710086636515
  12.8 0.027416273511
  25.6 0.00161138147606
  51.2 1.51410458608e-05
  102.4 -2.8165242665e-06
  O D
  0.669340 1.000000
  O F
  1.423104 1.000000
  O S
  0.2 1.0
  O S
  0.6000000000000001 1.0
  O P 
  0.2 1.0
  O P 
  0.6000000000000001 1.0
  '''),
'Sr':gto.basis.parse('''
  Sr S
  0.4 -1.12060079327
  0.8 0.667777052063
  1.6 -0.264303359332
  3.2 0.107893172552
  6.4 -0.0467092058199
  12.8 0.0190367620017
  25.6 -0.00656798101781
  51.2 0.0015551383896
  102.4 -0.000192791210535
  Sr P
  0.4 -0.465851775006
  0.8 0.226987549209
  1.6 -0.066094426312
  3.2 0.0162954710475
  6.4 -0.00365084618826
  12.8 0.000705426577439
  25.6 -0.00017848079055
  51.2 3.06496147379e-05
  102.4 -2.71611465382e-06
  Sr D
  0.528269 1.000000
  Sr S
  0.2 1.0
  Sr S
  0.6000000000000001 1.0
  Sr P
  0.2 1.0
  Sr P
  0.6000000000000001 1.0 
  ''')}

minbasis2={
'Cu':gto.basis.parse('''
  Cu S
  0.4 -0.0364413530365
  0.8 0.358763832032
  1.6 0.474343247928
  3.2 0.498447510592
  6.4 -0.270094109886
  12.8 -0.19207249392
  25.6 0.0629633270568
  51.2 -0.00853551842135
  102.4 0.000812242437275
  Cu P
  18.594006 0.003652
  14.303011 -0.037076
  5.63142 0.21712
  2.683011 0.440973
  1.258573 0.358723
  0.574084 0.114846
  0.20923 0.007316
  Cu D
  0.4 -0.117221524191
  0.8 0.420466340314
  1.6 0.166556512124
  3.2 0.31519384244
  6.4 0.201127125456
  12.8 0.128021225822
  25.6 0.00894474569825
  51.2 -0.00051734713442
  102.4 2.47330807683e-05
  '''),
'O':gto.basis.parse('''
  O S
  0.4 0.351884869096
  0.8 0.395141155066
  1.6 0.179458356356
  3.2 -0.0432052834993
  6.4 -0.105221770065
  12.8 0.0197823639914
  25.6 -0.00197386728468
  51.2 0.000387842127399
  102.4 -5.57569955803e-05
  O P 
  0.4 0.205849797904
  0.8 0.303195415412
  1.6 0.218836382326
  3.2 0.136227390096
  6.4 0.0710086636515
  12.8 0.027416273511
  25.6 0.00161138147606
  51.2 1.51410458608e-05
  102.4 -2.8165242665e-06
  '''),
'Sr':gto.basis.parse('''
  Sr S
  0.4 -1.12060079327
  0.8 0.667777052063
  1.6 -0.264303359332
  3.2 0.107893172552
  6.4 -0.0467092058199
  12.8 0.0190367620017
  25.6 -0.00656798101781
  51.2 0.0015551383896
  102.4 -0.000192791210535
  ''')}

basis_order = {'Cu':[0,0,1,2,3,0,0,1,1,2,2],'O':[0,1,2,3,0,0,1,1],'Sr':[0,1,2,0,0,1,1]}

###########################################################################################
#Methods

#USE: testing method
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

#USE: constructing OAOs
def calcOAO(cell,mf):
  '''
  input: 
    cell and scf (PBC SCF) objects from pyscf 
  output:
    Calculates 1RDM on orthog atomic orbitals, orthogonalized
    using Lowdin S^1/2
    Returns (rdm1up+rd1mdn, rdm1up-rdm1dn)
  '''

  s=mf.get_ovlp(cell)[0]
  s12=sp.linalg.sqrtm(s).real 
  rdm1=mf.make_rdm1(mf.mo_coeff,mf.mo_occ)
  oao_dmu=reduce(np.dot,(s12,rdm1[0][0],s12))
  oao_dmd=reduce(np.dot,(s12,rdm1[1][0],s12))
  oao_dmu+=oao_dmd
  oao_dmd=oao_dmu-2*oao_dmd
  
  #Printouts. nup+ndown and nup-ndown
  cuMin=[0,1,4,5,6,13,14,15,16,17]
  up=0
  dn=0
  for i in range(len(cuMin)):
    up+=oao_dmu[cuMin[i],cuMin[i]]
    dn+=oao_dmd[cuMin[i],cuMin[i]]
  print(up,dn)
  '''
  print("CELL: Tr(OAO_dens) - Tot ",np.trace(oao_dmu),np.trace(oao_dmd))
  print("CELL: Tr(OAO_dens) - Cu 1",np.trace(oao_dmu[:35,:35]),np.trace(oao_dmd[:35,:35]))
  cuMin=[0,1,4,5,6,13,14,15,16,17]
  up=0
  dn=0
  for i in range(len(cuMin)):
    up+=oao_dmu[cuMin[i],cuMin[i]]
    dn+=oao_dmd[cuMin[i],cuMin[i]]
  print("CELL: Tr(OAO_dens) MIN - Cu 1",up,dn)
  print("CELL: Tr(OAO_dens) - O 1 ",np.trace(oao_dmu[280:304,280:304]),np.trace(oao_dmd[280:304,280:304]))
  print("CELL: Tr(OAO_dens) - Sr 1",np.trace(oao_dmu[664:681,664:681]),np.trace(oao_dmd[664:681,664:681]))
  '''

  return (oao_dmu,oao_dmd)

#USE: plotting OAO occupations
def plotSubOAO(oao_dmu,oao_dmd):
  '''
  input: up and down density matrices on OAO basis
  plot sub-density matrices on different types of atoms
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
  #Construct the sub DM for oxygen
  for i in range(16):
    o_oao.append([oao_dmu[280+i*no:280+(i+1)*no,280+i*no:280+(i+1)*no],oao_dmd[280+i*no:280+(i+1)*no,280+i*no:280+(i+1)*no]])

  #Plot sub DM for oxygen
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

#USE: plotting OAOs
def moToOAO(mf):
  '''
    input: mf (PBC) objects for PYSCF
    output: mf objects, but with mf.mo_coeff the coefficients of the OAO relative to AOs
  '''
  s=mf.get_ovlp()[0]
  sm12=orth.lowdin(s) #S^{-1/2}
  mf.mo_coeff[0][0]=sm12
  mf.mo_coeff[1][0]=sm12
  return cell,mf

'''
def calcOAONewBas(cell,mf,pcell,pmf):
  #input: 
  #  cell and scf (PBC SCF) objects from pyscf, 
  #  and a new basis to represent RDM in 
  #output:
  #  Calculates 1RDM on orthog atomic orbitals defined by 
  #  the new basis, orthogonalized using Lowdin S^1/2
  #  Returns (rdm1up+rd1mdn, rdm1up-rdm1dn)

  #Edit MO coefficients on new basis
  from pyscf.pbc import gto as pbcgto
  s2 = pcell.pbc_intor('int1e_ovlp', hermi=1)               #Overlap in new basis
  s2inv = sp.linalg.inv(s2)                                 #Inverse of overlap in new basis
  sbar = pbcgto.cell.intor_cross('int1e_ovlp', cell, pcell) #Overlap between bases
  mo_coeff=np.zeros(mf.mo_coeff.shape)
  pmf.mo_coeff[0][0]=reduce(np.dot,(mf.mo_coeff[0][0],sbar,s2inv))
  pmf.mo_coeff[1][0]=reduce(np.dot,(mf.mo_coeff[1][0],sbar,s2inv))

  #Construct the RDMs on the new basis
  s12=sp.linalg.sqrtm(s2).real #Sqrt of overlap in new basis
  rdm1=mf.make_rdm1(pmf.mo_coeff,pmf.mo_occ)
  oao_dmu=reduce(np.dot,(s12,rdm1[0][0],s12))
  oao_dmd=reduce(np.dot,(s12,rdm1[1][0],s12))
  oao_dmu+=oao_dmd
  oao_dmd=oao_dmu-2*oao_dmd
  
  #Printouts. nup+ndown and nup-ndown
  print("CELL: Tr(OAO_dens) - Tot ",np.trace(oao_dmu),np.trace(oao_dmd))
  print("CELL: Tr(OAO_dens) - Cu 1",np.trace(oao_dmu[:35,:35]),np.trace(oao_dmd[:35,:35]))
  cuMin=[0,1,4,5,6,13,14,15,16,17]
  up=0
  dn=0
  for i in range(len(cuMin)):
    up+=oao_dmu[cuMin[i],cuMin[i]]
    dn+=oao_dmd[cuMin[i],cuMin[i]]
  print("CELL: Tr(OAO_dens) MIN - Cu 1",up,dn)
  print("CELL: Tr(OAO_dens) - O 1 ",np.trace(oao_dmu[280:304,280:304]),np.trace(oao_dmd[280:304,280:304]))
  print("CELL: Tr(OAO_dens) - Sr 1",np.trace(oao_dmu[664:681,664:681]),np.trace(oao_dmd[664:681,664:681]))
  

  return (oao_dmu,oao_dmd)
'''

def calcIAO(cell,mf,basis):
  '''
  input: 
    cell and scf (PBC SCF) objects from pyscf and basis
    to calculate IAOs on
  output:
    Calculates 1RDM on orthog atomic orbitals, orthogonalized
    using Lowdin S^1/2
    Returns (rdm1up+rd1mdn, rdm1up-rdm1dn)
  '''
  s=mf.get_ovlp()[0]

  mo_occ = mf.mo_coeff[0][0][:,mf.mo_occ[0][0]>0]
  a = lo.iao.iao(cell, mo_occ, minao=basis)
  # orthogonalize iao
  a = lo.vec_lowdin(a, s)
  # transform mo_occ to iao representation. note the ao dimension is reduced
  mo_occ = reduce(np.dot, (a.T, s, mo_occ))
  dm_u = np.dot(mo_occ, mo_occ.T)
  
  mo_occ = mf.mo_coeff[1][0][:,mf.mo_occ[1][0]>0]
  a = lo.iao.iao(cell, mo_occ, minao=basis)
  # orthogonalize iao
  a = lo.vec_lowdin(a, s)
  # transform mo_occ to iao representation. note the ao dimension is reduced
  mo_occ = reduce(np.dot, (a.T, s, mo_occ))
  dm_d = np.dot(mo_occ, mo_occ.T)
  
  dm_u+=dm_d
  dm_d=dm_u-2*dm_d
  
  #TESTING
  '''
  print("IAO, single Cu:", np.trace(dm_u[:9,:9]),np.trace(dm_d[:9,:9]))
  print("IAO, all Cu:",np.trace(dm_u[:72,:72]),np.trace(dm_d[:72,:72]))
  print("IAO, single O:",np.trace(dm_u[72:76,72:76]),np.trace(dm_d[72:76,72:76]))
  print("IAO, all O:",np.trace(dm_u[72:136,72:136]),np.trace(dm_d[72:136,72:136]))
  print("IAO, single Sr:",np.trace(dm_u[136:137,136:137]),np.trace(dm_d[136:137,136:137]))
  print("IAO, all Sr:",np.trace(dm_u[136:,136:]),np.trace(dm_d[136:,136:]))
  '''
  return (dm_u,dm_d)

#USE: plotting IAO occupations
def plotSubIAO(oao_dmu,oao_dmd):
  '''
  input: up and down density matrices on OAO basis
  plot sub-density matrices on different types of atoms
  '''
  ncu=9 #Number of basis elements per copper
  no=4  #Number of basis elements per oxygen
  nsr=1 #Number of basis elements per strontium
  
  cu_oao=[]
  #Construct the sub DM for copper
  for i in range(8):
    cu_oao.append([oao_dmu[i*ncu:(i+1)*ncu,i*ncu:(i+1)*ncu],oao_dmd[i*ncu:(i+1)*ncu,i*ncu:(i+1)*ncu]])
 
  #Plot sub DM for copper
  labels=['3s','3px','3py','3pz','3dxy','3dyz','3dz^2','3dxz','3dx^2-y^2']
  plt.suptitle("Cu OAO occupations")
  plt.subplot(121)
  plt.title("Nup+Ndown") 
  plt.xticks(np.arange(0,ncu,1),labels,rotation='vertical')
  for i in range(8):
    plt.plot(np.diag(cu_oao[i][0]),'o')
  plt.subplot(122)
  plt.title("Nup-Ndown") 
  plt.xticks(np.arange(0,ncu,1),labels,rotation='vertical')
  for i in range(8):
    plt.plot(np.diag(cu_oao[i][1]),'o')
  plt.show()

  o_oao=[]
  #Construct the sub DM for oxygen
  for i in range(16):
    o_oao.append([oao_dmu[72+i*no:72+(i+1)*no,72+i*no:72+(i+1)*no],oao_dmd[72+i*no:72+(i+1)*no,72+i*no:72+(i+1)*no]])
  
  #Plot sub DM for oxygen
  labels=['2s','2px','2py','2pz']
  plt.suptitle("O OAO occupations")
  plt.subplot(121)
  plt.title("Nup+Ndown") 
  plt.xticks(np.arange(0,no,1),labels,rotation='vertical')
  for i in range(16):
    plt.plot(np.diag(o_oao[i][0]),'o')
  plt.subplot(122)
  plt.title("Nup-Ndown") 
  plt.xticks(np.arange(0,no,1),labels,rotation='vertical')
  for i in range(16):
    plt.plot(np.diag(o_oao[i][1]),'o')
  plt.show()

###########################################################################################
#Run
direc="CHK"

#READIN
cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=direc+"/GRED.DAT",kred=direc+"/KRED.DAT",cryoutfn=direc+"/prop.in.o")

#OAOs 
#oao_dmu,oao_dmd=calcOAO(cell,mf)
#iao_dmu,iao_dmd=calcIAO(cell,mf,minbasis)
iao_dmu,iao_dmd=calcIAO(cell,mf,minbasis2)
plotSubIAO(iao_dmu,iao_dmd)
