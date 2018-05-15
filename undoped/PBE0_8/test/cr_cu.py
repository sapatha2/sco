#PYSCF copper calculation 
from pyscf import gto, scf, lo
import numpy as np 
from functools import reduce 
from crystal2pyscf import crystal2pyscf_mol, crystal2pyscf_cell

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
  0.6 1.0''')}

##############################################################
#WITH BASIS ORDER: GOOD!
basis_order = {'Cu':[0,0,1,2,3,0,0,1,1,2,2],'O':[0,1,2,3,0,0,1,1],'Sr':[0,1,2,0,0,1,1]}
mol,mf=crystal2pyscf_mol(basis=basis,basis_order=basis_order)

s=mf.get_ovlp()
mo_up=mf.mo_coeff[0]
mo_dn=mf.mo_coeff[1]

#MO OVERLAPS
mo_ovlpu=reduce(np.dot,(mo_up.T,s,mo_up))
mo_ovlpd=reduce(np.dot,(mo_dn.T,s,mo_dn))
print(np.trace(mo_ovlpu),np.trace(mo_ovlpd))

#1RDM ON MO
rdm1=mf.make_rdm1(mf.mo_coeff,mf.mo_occ)
mo_dmu=reduce(np.dot,(mo_up.T,s,rdm1[0],s,mo_up))
mo_dmd=reduce(np.dot,(mo_dn.T,s,rdm1[1],s,mo_dn))
print(np.trace(mo_dmu),np.trace(mo_dmd))
