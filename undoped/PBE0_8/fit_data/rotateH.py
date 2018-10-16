from crystal2pyscf import crystal2pyscf_mol, crystal2pyscf_cell
from pyscf import gto, scf, lo
from pyscf.lo import orth
from pyscf.lo.iao import iao
from functools import reduce
import numpy as np 
import matplotlib.pyplot as plt
import pickle
from pyscf import fci
from downfold_tools import gen_slater_tbdm

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

minbasis={
'Cu':gto.basis.parse('''
Cu S  
  0.2 0.0286878866897
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
 0.2 2.20172386547
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
  0.2 0.279289473482
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
  0.2 0.215448374976
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
  0.2 0.283073445696
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
0.2 0.688519195726
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
def calcIAO(cell,mf,basis,occ):
  '''
  input: 
    cell and scf (PBC SCF) objects from pyscf and basis
    to calculate IAOs on
    occupied MOs which IAOs should span
  output:
    Calculates 1RDM on orthog atomic orbitals, orthogonalized
    using Lowdin S^1/2
    Returns coefficient matrix for IAOs 
  '''
  s=mf.get_ovlp()[0]

  mo_occ = mf.mo_coeff[0][0][:,occ]
  mo_occ2 = mf.mo_coeff[1][0][:,occ]
  mo_occ=np.concatenate((mo_occ,mo_occ2),axis=1)
  a = lo.iao.iao(cell, mo_occ, minao=basis)
  a = lo.vec_lowdin(a, s)
 
  return a

#IAO DM
def rdmIAO(mf,a):
  '''
  input:
    mf object for calculation
  output:
    1rdm for spin up and down 
  '''
  s=mf.get_ovlp()[0]

  mo_occ = mf.mo_coeff[0][0][:,mf.mo_occ[0][0]>0]
  mo_occ = reduce(np.dot, (a.T, s, mo_occ))
  dm_u = np.dot(mo_occ, mo_occ.T)
  
  mo_occ = mf.mo_coeff[1][0][:,mf.mo_occ[1][0]>0]
  mo_occ = reduce(np.dot, (a.T, s, mo_occ))
  dm_d = np.dot(mo_occ, mo_occ.T)

  return np.array([dm_u,dm_d])


#IAO Matrix
def matIAO(mat,mf,a):
  return reduce(np.dot(t,mat,t.T))

#IAO 2RDM
def rdm2diag(obdm):
  '''THIS MIGHT BE ONLY
  VALID IF OBDM IS ON THE 
  OCCUPIED STATES, NOT THE IAOS?
  '''
  uu=np.zeros(obdm.shape[1])
  dd=np.zeros(obdm.shape[1])
  ud=np.diag(obdm[0])*np.diag(obdm[1])
  return (uu,dd,ud)

###########################################################################################
#Run
ch=[] #Hybridized hopping (Cu d and sigma, no curl)
cc=[] #Curly hopping
cxy=[] #x hopping
nnh=[[0,6,8,11],
[1,7,8,9],
[2,4,9,15],
[3,5,10,11],
[0,4,12,15],
[1,5,12,13],
[2,6,13,14],
[3,7,14,15]] #Nearest neighbors oxygens for a given copper
signh=[[-1,1,1,-1],
[-1,1,-1,1],
[-1,1,-1,1],
[-1,1,-1,1],
[1,-1,1,-1],
[1,-1,-1,1],
[1,-1,1,-1],
[1,-1,-1,1]] #Hopping signs for neighbor oxygens

nnhc=[[15,12,8,11],
[12,13,9,8],
[13,14,10,9],
[14,15,11,10],
[9,10,12,15],
[10,11,13,12],
[11,8,14,13],
[8,9,15,14],
[0,1,7,6],
[1,2,4,7],
[2,3,5,4],
[3,0,6,5],
[4,5,1,0],
[5,6,2,1],
[6,7,3,2],
[7,4,0,3]]
sgn=[1,-1,1,-1]

########################
#CREATE IAO 
direc="../CHK"
occ=[i for i in range(132)] + [32,33,34,36]  #Have to include the virtual dx2-y2 orbitals
cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=direc+"/GRED.DAT",kred=direc+"/KRED.DAT",cryoutfn=direc+"/prop.in.o")
a=calcIAO(cell,mf,minbasis,occ)

#######################
#CALCULATE E, 1RDM
direclist=["CHK","FLP","COL","BCOL","BLK","ACHN","FM"]
Elist=[-1.8424749919822E+03,-1.8424644227349E+03,-1.8424527681040E+03,
-1.8424548946115E+03,-1.8424530584090E+03,-1.8424527307933E+03,
-1.8424227577357E+03]

for i in range(len(direclist)):
  #Base states
  direc="../"+direclist[i]
  E=Elist[i]
  cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=direc+"/GRED.DAT",kred=direc+"/KRED.DAT",cryoutfn=direc+"/prop.in.o")

  s=mf.get_ovlp()[0]
 
  #FULL MATRIX
  #H1u=np.diag(mf.mo_energy[0][0])*27.2114
  #H1d=np.diag(mf.mo_energy[1][0])*27.2114

  #IAO SPACE SPAN 
  mf.mo_occ[0][0][132:135]=1
  mf.mo_occ[0][0][136]=1
  mf.mo_occ[1][0][132:135]=1
  mf.mo_occ[1][0][136]=1
  H1u=np.diag(mf.mo_energy[0][0])*27.2114
  H1d=np.diag(mf.mo_energy[1][0])*27.2114

  #OCCUPIED ONLY 
  H1u=np.diag(mf.mo_occ[0][0]*mf.mo_energy[0][0])*27.2114
  H1d=np.diag(mf.mo_occ[1][0]*mf.mo_energy[1][0])*27.2114

  e1u=reduce(np.dot,(a.T,s,mf.mo_coeff[0][0],H1u,mf.mo_coeff[0][0].T,s.T,a))
  e1d=reduce(np.dot,(a.T,s,mf.mo_coeff[1][0],H1d,mf.mo_coeff[1][0].T,s.T,a))

  ncu=10
  no=4
  nsr=1
 
  print(-e1u[ncu*0+9,ncu*8+no*0+1]-e1d[ncu*0+9,ncu*8+no*0+1]) #sigma
  print(-e1u[ncu*8+no*0+2,ncu*8+no*12+1]-e1d[ncu*8+no*0+2,ncu*8+no*12+1]) #pi
  print(e1u[ncu*8+no*0+1,ncu*8+no*12+1]+e1d[ncu*8+no*0+1,ncu*8+no*12+1]) #xy

