from crystal2pyscf import crystal2pyscf_mol
#from pyscf.scf.uhf import make_rdm1
from pyscf.scf.hf import make_rdm1
from pyscf.scf.hf import get_ovlp
from pyscf import gto, scf, lo
import numpy as np 
from pyscf.lo import orth
from functools import reduce
import scipy as sp 

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

########################################################
#1rdm, atomic orbitals
mol,mf=crystal2pyscf_mol(basis=basis)

#METHOD 0
mo_occ = mf.mo_coeff[0][:,mf.mo_occ[0]>0]
s12=sp.linalg.sqrtm(mf.get_ovlp())
mo_occ=np.dot(mo_occ.T,s12)
dm=np.dot(mo_occ.T,mo_occ)
print(np.trace(dm))

#METHOD 1 - SAME RESULTS AS METHOD 0
#BASED ON EXAMPLE HERE: https://github.com/sunqm/pyscf/blob/627afd3072eee567c6ff0805ecdb5132fbc33a3b/examples/local_orb/02-pop_with_iao.py
mo_occ = mf.mo_coeff[0][:,mf.mo_occ[0]>0]
a=np.identity(len(mo_occ))
a = lo.vec_lowdin(a, mf.get_ovlp())
mo_occ = reduce(np.dot, (a.T, mf.get_ovlp(), mo_occ))
dm = np.dot(mo_occ, mo_occ.T)
print(np.trace(dm))

#METHOD 2 - SAME RESULTS AS METHOD 1
#BASED ON EXAMPLE HERE:  https://github.com/sunqm/pyscf/blob/627afd3072eee567c6ff0805ecdb5132fbc33a3b/examples/local_orb/02-pop_with_iao.py
C = lo.orth_ao(mf, 'lowdin')
mo = np.linalg.solve(C, mf.mo_coeff[0])
dm = make_rdm1(mo, mf.mo_occ[0])
print(np.trace(dm))


