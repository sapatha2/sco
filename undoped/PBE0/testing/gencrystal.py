import pymatgen as mg
from Crystal import CrystalWriter
from pymatgen.io.cif import CifParser

'''
#Write CIF file
#Physica C 181 ( 1991 ) 206-208 North-Holland
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
fname="SrCuO2_221sc.cif"
myfile=open(fname,"r")
write.set_struct_fromcif(myfile.read(),primitive=False)

#write.xml_name='../../../../../../../Documents/Research/GitHub/bus_autogenv2/BFD_Library.xml'
#write.xml_name='../../../../../../../Documents/Research/GitHub/bus_autogenv2/BFD_PBC0.20.xml'
write.xml_name='../../../../../../../Documents/Research/GitHub/bus_autogenv2/BFD_PBC.xml'
write.cutoff=0.15

write.functional={'exchange':'PBE','correlation':'PBE','hybrid':25,'predefined':None}
write.kmesh=[6,6,4]
write.gmesh=12

write.write_crys_input("crys_BFD_PBC.in")

#After this, still need to edit calculation properties and symmetries
