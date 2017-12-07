import cubetools
nup=68
ndown=64
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
up['data']-=down['data']

#CT state
down_ct=cubetools.read_cube("qwalk_000.plot_d_ct1.dens.cube")
down_ct=cubetools.normalize_abs(down_ct)
down_ct['data']*=ndown
up['data']-=down_ct['data'] #make 'up' the spin density
cubetools.write_cube(up,"qwalk_000.ct1_spin.dens.cube")
up['data']+=2*down_ct['data'] #make 'up' the charge density
cubetools.write_cube(up,"qwalk_000.ct1_chg.dens.cube")
up['data']-=down_ct['data']

down_ct=cubetools.read_cube("qwalk_000.plot_d_ct2.dens.cube")
down_ct=cubetools.normalize_abs(down_ct)
down_ct2=cubetools.read_cube("qwalk_000.plot_d_ct3.dens.cube")
down_ct2=cubetools.normalize_abs(down_ct2)
down_ct['data']*=ndown
down_ct2['data']*=ndown
up['data']-=(down_ct['data']+down_ct2['data'])/2 #make 'up' the spin density
cubetools.write_cube(up,"qwalk_000.ct2_spin.dens.cube")
up['data']+=(down_ct['data']+down_ct['data']) #make 'up' the charge density
cubetools.write_cube(up,"qwalk_000.ct2_chg.dens.cube")
