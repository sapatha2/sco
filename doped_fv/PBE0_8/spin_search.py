#Search the spins configurations on an 8 site calculation 
import numpy as np 
import matplotlib.pyplot as plt

NN=[[1,3,4,6],[0,2,5,7],[1,3,4,6],[0,2,5,7],[0,2,5,7],[1,3,4,6],[0,2,5,7],[1,3,4,6]]
sigJs=[]
locJs=[]
Nsite=8

#TABLE OF UNIQUE STATES
'''
#STATE      SIGJ     LOCJ    COMPLETE
-------------------------------------
CHK         -4       T       Y
FLP         -2       T       Y
ACHN         0       T       Y
COL          0       F       Y
BLK          0       F       Y
BCOL         0       F       Y
FM           4       T       Y
-------------------------------------
TETR(SWT)   -1       T       N
RECT3(CFLP)  0       T       N
ELBW         0       T       N
DIAG         0       T       N
BBCOL        0       F       N
RECT2        1       T       N
BCHK         2       T       N
-------------------------------------
'''
#CHK, FLP, TETR, ACHN, RECT3, ELBW, COL, DIAG, BBCOL, BLK, BCOL, RECT2, BCHK, FM
for i in [165,88,120,95,13,52,85,18,23,51,147,9,32,255]:
  spin=format(i,'0'+str(Nsite)+'b')
  spin=[float(spin[j])-0.5 for j in range(len(spin))]

  sigJ=0
  locJ=False
  for j in range(Nsite):
    nnterm=0
    for k in range(4):
      sigJ+=spin[j]*spin[NN[j][k]]
      nnterm+=spin[NN[j][k]]
    if(nnterm!=0): locJ=True
  sigJ/=2.
  Mat=[[spin[2],spin[6],spin[0],spin[4],spin[2],spin[6]],
       [spin[1],spin[5],spin[3],spin[7],spin[1],spin[5]],
       [spin[0],spin[4],spin[2],spin[6],spin[0],spin[4]],
       [spin[3],spin[7],spin[1],spin[5],spin[3],spin[7]],
       [spin[2],spin[6],spin[0],spin[4],spin[2],spin[6]]]
  plt.matshow(Mat)
  plt.title(str(sigJ)+","+str(locJ))
  plt.show()
  sigJs.append(sigJ)
  locJs.append(locJ)

sigJs=np.array(sigJs)
locJs=np.array(locJs)
