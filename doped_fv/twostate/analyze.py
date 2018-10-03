#Actual analysis
import json
from func import iao1rdm
from crystal2pyscf import crystal2pyscf_cell
import numpy as np

########################################################
#GET IAO 1RDM
basis=json.load(open("basis.json","r"))
basis_order = {'Cu':[0,0,1,2,3,0,0,1,1,2,2],'O':[0,1,2,3,0,0,1,1],'Sr':[0,1,2,0,0,1,1]}

direc1='../../undoped/PBE0_8/CHK'
cell,mf=crystal2pyscf_cell(totspin=0.0,netcharge=0.0,nelectron=264,basis=basis,basis_order=basis_order,gred=direc1+"/GRED.DAT",kred=direc1+"/KRED.DAT")

'''
state="FLP1"
direc1="../PBE0_8/"+state
cell,mf=crystal2pyscf_cell(totspin=1.0,netcharge=1.0,nelectron=265,basis=basis,basis_order=basis_order,gred=direc1+"/GRED.DAT",kred=direc1+"/KRED.DAT")
'''

mf.xc='PBE0' #This is not set in crystal2pyscf_cell
print(cell.nelec)
print(cell.mesh)
mf.e_tot=mf.energy_tot()
print("Total energy:", mf.e_tot)

#minbasis2=json.load(open("minbasis2.json","r"))
#iao_dmu,iao_dmd=iao1rdm(cell,mf,minbasis2)
#data={'E':mf.e_tot,'iao_dmu':iao_dmu.tolist(),'iao_dmd':iao_dmd.tolist()}
#json.dump(data,open(state+".dat","w"))

########################################################
#ANALYZE IAO 1RDM
'''
print("1RDM trace: ",np.trace(iao_dmu),np.trace(iao_dmd))
print(iao_dmu.shape,iao_dmd.shape)

#1rdm = 152 elements = (8*10) + (16*4) + (8*1)
culabels=['3s','4s','3px','3py','3pz','3dxy','3dyz','3dz^2','3dxz','3dx^2-y^2']
olabels=['2s','2px','2py','2pz']
srlables=['5s']
ncu=8
no=16
nsr=8
'''
