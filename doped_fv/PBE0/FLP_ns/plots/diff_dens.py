import cubetools
import copy 

s1=cubetools.read_cube("qwalk_000.spin.dens.cube")
s2=cubetools.read_cube("../../../../undoped/PBE0/FLP/plots/qwalk_000.spin.dens.cube")

c1=cubetools.read_cube("qwalk_000.chg.dens.cube")
c2=cubetools.read_cube("../../../../undoped/PBE0/FLP/plots/qwalk_000.chg.dens.cube")

s1['data']-=s2['data'] #make 'up' the spin density
cubetools.write_cube(s1,"qwalk_000.spin_diff.dens.cube")
c1['data']-=c2['data'] #make 'up' the charge density
cubetools.write_cube(c1,"qwalk_000.chg_diff.dens.cube")

