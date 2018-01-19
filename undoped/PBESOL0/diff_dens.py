import cubetools
AFMc=cubetools.read_cube("AFM_small/qwalk_000.chg.dens.cube")
AFMs=cubetools.read_cube("AFM_small/qwalk_000.spin.dens.cube")
FMc=cubetools.read_cube("FM_small/qwalk_000.chg.dens.cube")
FMs=cubetools.read_cube("FM_small/qwalk_000.spin.dens.cube")
AFM_CTc=cubetools.read_cube("AFM_small/qwalk_000.ct.chg.dens.cube")
AFM_CTs=cubetools.read_cube("AFM_small/qwalk_000.ct.spin.dens.cube")

#These are properly normalized
cubetools.normalize_abs(AFMc)
cubetools.normalize_abs(FMc)
cubetools.normalize_abs(AFM_CTc)

FMc['data']-=AFMc['data']
FMs['data']-=AFMs['data']
AFM_CTc['data']-=AFMc['data']
AFM_CTs['data']-=AFMs['data']
'''
cubetools.write_cube(FMc,"qwalk_000.fm_diff.chg.dens.cube")
cubetools.write_cube(FMs,"qwalk_000.fm_diff.spin.dens.cube")
cubetools.write_cube(AFM_CTc,"qwalk_00.afm_ct_diff.chg.dens.cube")
cubetools.write_cube(AFM_CTs,"qwalk_000.afm_ct_diff.spin.dens.cube")
'''
cubetools.normalize_abs(FMc)
cubetools.normalize_abs(AFM_CTc)
cubetools.normalize_abs(FMs)
cubetools.normalize_abs(AFM_CTs)
