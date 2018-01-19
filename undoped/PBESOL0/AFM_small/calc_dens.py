import cubetools
nup=66
ndown=66
import copy 

'''
up=cubetools.read_cube("qwalk_000.plot_u.dens.cube")
down=cubetools.read_cube("qwalk_000.plot_d.dens.cube")
up=cubetools.normalize_abs(up)
down=cubetools.normalize_abs(down)
up['data']*=nup
down['data']*=ndown
up['data']-=down['data'] #make 'up' the spin density
cubetools.write_cube(up,"qwalk_000.spin.dens.cube")
up['data']+=2*down['data'] #make 'up' the charge density
cubetools.write_cube(up,"qwalk_000.chg.dens.cube")
'''
up=cubetools.read_cube("qwalk_000.plot_u.dens.cube")
down=cubetools.read_cube("qwalk_000.plot_d.dens.cube")
up_ct=cubetools.read_cube("qwalk_000.plot_u_ct.dens.cube")
down_ct=cubetools.read_cube("qwalk_000.plot_d_ct.dens.cube")
up=cubetools.normalize_abs(up)
down=cubetools.normalize_abs(down)
up_ct=cubetools.normalize_abs(up_ct)
down_ct=cubetools.normalize_abs(down_ct)
up['data']*=nup
down['data']*=ndown
up_ct['data']*=nup
down_ct['data']*=ndown

#Individual CT determinants
c_u_ct=copy.deepcopy(up_ct)
s_u_ct=copy.deepcopy(up_ct)
c_d_ct=copy.deepcopy(up)
s_d_ct=copy.deepcopy(up)
c_u_ct['data']+=down['data']
s_u_ct['data']-=down['data']
c_d_ct['data']+=down_ct['data']
s_d_ct['data']-=down_ct['data']

#Full CT state (densities add regardless of the sign between the two determinants)
c_u_ct['data']+=c_d_ct['data']
c_u_ct['data']/=2
s_u_ct['data']+=s_d_ct['data']
s_u_ct['data']/=2

cubetools.write_cube(c_u_ct,"qwalk_000.ct.chg.dens.cube")
cubetools.write_cube(s_u_ct,"qwalk_000.ct.spin.dens.cube")

