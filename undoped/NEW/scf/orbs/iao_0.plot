method {
  plot
  orbitals {
  cutoff_mo
    magnify 1
    nmo 76
    orbfile iao_0.orb
    include iao.basis
    centers { useglobal }
  }
  PLOTORBITALS { 
   11 12 13 14 15 16 17 18 19 20 
  }
}
include iao_0.sys 
