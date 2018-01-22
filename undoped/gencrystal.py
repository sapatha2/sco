import pymatgen as mg
from Crystal import CrystalWriter
from pymatgen.io.cif import CifParser

'''
#Write CIF file
lattice=mg.Lattice.tetragonal(7.8558,3.4338)
structure=mg.Structure(lattice,["Cu","Cu","Cu","Cu",
"O","O","O","O","O","O","O","O",
"Sr","Sr","Sr","Sr"],
[[0.0,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0],[0.5,0.5,0.0],
[0.25,0.0,0.0],[0.75,0.0,0.0],[0.25,0.5,0.0],[0.75,0.5,0.0],[0.0,0.25,0.0],[0.0,0.75,0.0],[0.5,0.25,0],[0.5,0.75,0.0],
[0.25,0.25,0.5],[0.25,0.75,0.5],[0.75,0.25,0.5],[0.75,0.75,0.5]])
structure.to(filename="SrCuO2_221sc.cif")
'''

#Read CIF file and write output
write=CrystalWriter()
write.xml_name='../../autogenv2/BFD_Library.xml'
fname="SrCuO2_221sc.cif"
myfile=open(fname,"r")
write.set_struct_fromcif(myfile.read(),primitive=False)
write.write_crys_input("crys_pbe0.in")

#Need to add
'''
SYMMETRY GROUP
SUPERCON (vs SUPERCEL)
MODISYMM
Make sure basis cutoff is okay
SHRINK 
MAXCYCL
ATOMSPIN
'''


