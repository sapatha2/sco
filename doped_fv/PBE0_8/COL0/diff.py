import numpy as np 
import matplotlib.pyplot as plt 

#Charge and spin Mulliken populations
COLs=np.array([[0,-0.6442758,-0.6442758,0],
      [0.6442758,0.6442758,0.6442758,0.6442758],
      [0,-0.6442759,-0.6442759,0]])
COLc=np.array([[0,18.0003385,18.0003385,0],
      [18.0003362,18.0003382,18.0003382,18.0003362],
      [0,18.0003386,18.0003386,0]])
COL0s=np.array([[0,-0.4154364,-0.4154367,0],
       [0.4154353,0.4154368,0.4154368,0.4154350],
       [0,-0.4154358,-0.4154356,0]])
COL0c=np.array([[0,18.1292690,18.1292669,0],
       [18.1292643,18.1292696,18.1292684,18.1292620],
       [0,18.1292637,18.1292624,0]])

diffc=COL0c-COLc
diffs=COL0s-COLs

print(diffs)
plt.matshow(diffs)
plt.colorbar()

plt.show()
