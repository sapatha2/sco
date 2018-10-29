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
import find_connect 

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

########################
#RUN IAO 
direc="../CHK"
occ=[i for i in range(132)] + [32,33,34,36]  #Have to include the virtual dx2-y2 orbitals
cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=direc+"/GRED.DAT",kred=direc+"/KRED.DAT",cryoutfn=direc+"/prop.in.o")
a=calcIAO(cell,mf,minbasis,occ)

direclist=["CHK","FLP","COL","BCOL","BLK","ACHN","FM"]
culabels=["3s","4s","3px","3py","3pz",'3dxy','3dyz','3dz2','3dxz','3dx2y2']
oxlabels=["2s","2px","2py","2pz"]
srlabels=["5s"]
orig_labels=[]
#labels=(culabels)*8+(oxlabels)*16+(srlabels)*8
for i in range(8):
  orig_labels+=[x+str(i+1) for x in culabels]
for i in range(16):
  orig_labels+=[x+str(i+1) for x in oxlabels]
for i in range(8):
  orig_labels+=[x+str(i+1) for x in srlabels]
orig_labels=np.array(orig_labels)

for i in range(len(direclist)):
  #Base states
  direc="../"+direclist[i]
  cell,mf=crystal2pyscf_cell(basis=basis,basis_order=basis_order,gred=direc+"/GRED.DAT",kred=direc+"/KRED.DAT",cryoutfn=direc+"/prop.in.o")

  s=mf.get_ovlp()[0]
  H1u=np.diag(mf.mo_energy[0][0])
  H1d=np.diag(mf.mo_energy[1][0])
  e1u=27.2114*reduce(np.dot,(a.T,s,mf.mo_coeff[0][0],H1u,mf.mo_coeff[0][0].T,s.T,a))
  e1d=27.2114*reduce(np.dot,(a.T,s,mf.mo_coeff[1][0],H1d,mf.mo_coeff[1][0].T,s.T,a))
  e1u=(e1u+e1u.T)/2
  e1d=(e1d+e1d.T)/2
  
  #PLOT EIGENVALUES
  '''
  w,vr=np.linalg.eigh(e1u)
  plt.subplot(221)
  plt.plot(mf.mo_energy[0][0][:len(w)]*27.2112,'go',label='DFT PBE0')
  plt.plot(w,'bo',label='rotated')
  plt.ylabel("E (eV)")
  plt.xlabel("Eigenvalue")
  plt.legend(loc=2)
  
  plt.subplot(223)
  plt.plot(np.arange(126,137),mf.mo_energy[0][0][126:137]*27.2112,'go',label='DFT PBE0')
  plt.plot(np.arange(126,137),w[126:137],'bo',label='rotated')
  plt.ylabel("E (eV)")
  plt.xlabel("Eigenvalue")
  plt.legend(loc=2)
  
  w,vr=np.linalg.eigh(e1d)
  plt.subplot(222)
  plt.plot(mf.mo_energy[1][0][:len(w)]*27.2112,'go',w*27.2112,'bo')
  plt.ylabel("E (eV)")
  plt.xlabel("Eigenvalue")
  w,vr=np.linalg.eigh(e1u)
  
  plt.subplot(224)
  plt.plot(np.arange(126,137),mf.mo_energy[1][0][126:137]*27.2112,'go',label='DFT PBE0')
  plt.plot(np.arange(126,137),w[126:137],'bo',label='rotated')
  plt.ylabel("E (eV)")
  plt.xlabel("Eigenvalue")
 
  plt.show()
  '''
  
  #REMOVE CORE AND Z SPACE 
  relcu=[]
  for i in range(8):
    relcu+=[i*10+1,i*10+5,i*10+7,i*10+9]
  relo=[]
  for i in range(16):
    relo+=[80+i*4,80+i*4+1,80+i*4+2]
  
  rel=relcu+relo
  e1u=e1u[rel][:,rel]
  e1d=e1d[rel][:,rel]
  labels=orig_labels[rel]    
  print(labels)

  #SPIN UP 
  print(direc)
  #ordering = find_connect.recursive_order(e1u,[1e-10,1e-4,0.5,0.7,1,3,10,50,100,200])
  ordering = find_connect.recursive_order(e1u,[1e-10,1e-4,0.5,1,2,2.5,3,4,5,8])
  rearrange = e1u[ordering][:,ordering]

  fig=plt.figure()
  ax=fig.add_subplot(111)
  cax=ax.matshow(rearrange,vmin=-8.15,vmax=8.15,cmap='BrBG')
  #cax=ax.matshow(rearrange,vmin=-4.0,vmax=4.0,cmap='BrBG')
  fig.colorbar(cax)
  plt.title(direc+", spin up")

  ax.set_xticks(np.arange(len(labels)))
  ax.set_yticks(np.arange(len(labels)))
  ax.set_xticklabels(labels[ordering],rotation=90,fontsize=6)
  ax.set_yticklabels(labels[ordering],fontsize=6)
  #print(labels[ordering][:48])
  #print(labels[ordering][48:])
  plt.show()

  #ordering = find_connect.recursive_order(e1d,[1e-10,1e-4,0.5,0.7,1,3,10,50,100,200])
  ordering = find_connect.recursive_order(e1d,[1e-10,1e-4,0.5,1,2,2.5,3,4,5,8])
  rearrange = e1d[ordering][:,ordering]

  fig=plt.figure()
  ax=fig.add_subplot(111)
  cax=ax.matshow(rearrange,vmin=-8.15,vmax=8.15,cmap='BrBG')
  #cax=ax.matshow(rearrange,vmin=-4.0,vmax=4.0,cmap='BrBG')
  fig.colorbar(cax)
  plt.title(direc+", spin dn")
  
  ax.set_xticks(np.arange(len(labels)))
  ax.set_yticks(np.arange(len(labels)))
  ax.set_xticklabels(labels[ordering],rotation=90,fontsize=6)
  ax.set_yticklabels(labels[ordering],fontsize=6)
  #print(labels[ordering][:48])
  #print(labels[ordering][48:])
  plt.show()
