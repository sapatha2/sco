method { 
plot
orbitals {
cutoff_mo
  magnify 1
  nmo 933
  orbfile qwalk_000.orb
  include qwalk.basis
  centers { useglobal }
}
plotorbitals {
1 2 3 4 5 6 7 8 9 10
}
}
include qwalk_000.sys
