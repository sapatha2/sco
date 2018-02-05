import cubetools
nup=67
ndown=66
import copy 

#up=cubetools.read_cube("qwalk_000.plot_u.dens.cube")
#down=cubetools.read_cube("qwalk_000.plot_d.dens.cube")
#down=cubetools.read_cube("qwalk_000.plot_d_ex.dens.cube")
up=cubetools.read_cube("qwalk_110.plot_u.dens.cube")
down=cubetools.read_cube("qwalk_110.plot_d.dens.cube")
#down=cubetools.read_cube("qwalk_100.plot_d_ex.dens.cube")
up=cubetools.normalize_abs(up)
down=cubetools.normalize_abs(down)
up['data']*=nup
down['data']*=ndown
up['data']-=down['data'] #make 'up' the spin density
cubetools.write_cube(up,"qwalk_110.spin.dens.cube")
up['data']+=2*down['data'] #make 'up' the charge density
cubetools.write_cube(up,"qwalk_110.chg.dens.cube")

