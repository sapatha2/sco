method { 
plot
orbitals {
cutoff_mo
  magnify 1
  nmo 934
  orbfile qwalk_000.orb
  include qwalk.basis
  centers { useglobal }
}
plotorbitals {
    930 931   932   933   934  
}
}
include qwalk_000.sys
