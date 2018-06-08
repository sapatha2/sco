import cubetools
nup=134
ndown=130
import copy 

up=cubetools.read_cube("qwalk_000.up.plot.dens.cube")
down=cubetools.read_cube("qwalk_000.dn.plot.dens.cube")
up=cubetools.normalize_abs(up)
down=cubetools.normalize_abs(down)
up['data']*=nup
down['data']*=ndown
up['data']-=down['data'] #make 'up' the spin density
cubetools.write_cube(up,"qwalk_000.spin.dens.cube")
up['data']+=2*down['data'] #make 'up' the spin density
cubetools.write_cube(up,"qwalk_000.chg.dens.cube")
