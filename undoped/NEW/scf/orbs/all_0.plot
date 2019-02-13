method {
  plot
  orbitals {
  cutoff_mo
    magnify 1
    nmo 268
    orbfile all_0.orb
    include all.basis
    centers { useglobal }
  }
  PLOTORBITALS { 
    66  67  133 134 
    200 201 267 268
  }
}
include all_0.sys 
