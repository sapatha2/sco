import cubetools
nup=68
ndown=64
import copy 

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
