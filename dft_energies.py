import pickle
import numpy as np
import pandas as pd
data={
  #Ordering: x=0, x=0.25_fv, x=0.25
  'PBE0':{
    'S':{
      'x':['0','0.25_fv','0.25'],
      'CHK':np.array([-9.2123749620223E+02,-9.2092111151634E+02,-9.2092791881441E+02]),
      'COL':np.array([-9.2122638412999E+02,-9.2091969201656E+02,-9.2092670972882E+02]),
      'FLP1':np.array([-9.2122636552756E+02,-9.2092669006119E+02,-9.2093343498725E+02]),
      'FLP3':np.array([-9.2122636552756E+02,-9.2091219092014E+02,-9.2091978045593E+02]),
      'FM':np.array([-9.2121137910381E+02,-9.2091264058011E+02,-9.2092026703934E+02])
    },
    'NS':{
      'x':['0','0.25_fv','0.25'],
      'CHK':np.array([-9.2123749620223E+02,-9.2092669002320E+02,-9.2093343507729E+02]),
      'COL':np.array([-9.2122638412999E+02,-9.2092063265454E+02,-9.2092758720044E+02]),
      'FLP1':np.array([-9.2122636552756E+02,-9.2092669022287E+02,-9.2093343503482E+02]),
      'FLP3':np.array([-9.2122636552756E+02,-9.2091219109429E+02,-9.2091978048822E+02]),
      'FM':np.array([-9.2121137910381E+02,-9.2091264054414E+02,-9.2092002873599E+02])
    },
  },
  'HSE06':{
    'S':{
      'x':['0','0.25_fv','0.25'],
      'CHK':np.array([-9.2122881518088E+02,0,0]),
      'COL':np.array([-9.2121750188336E+02,0,0]),
      'FLP1':np.array([-9.2121759082240E+02,0,0]),
      'FLP3':np.array([-9.2121759082240E+02,0,0]),
      'FM':np.array([-9.2120231036158E+02,0,0])
    },
    'NS':{
      'x':['0','0.25_fv','0.25'],
      'CHK':np.array([-9.2122881518088E+02,0,0]),
      'COL':np.array([-9.2121750188336E+02,0,0]),
      'FLP1':np.array([-9.2121759082240E+02,0,0]),
      'FLP3':np.array([-9.2121759082240E+02,0,0]),
      'FM':np.array([-9.2120231036158E+02,0,0])
    },
  }
}
  
df=pd.DataFrame(data=data)
pickle.dump(df,open('dft_energies.p','wb'))

import matplotlib.pyplot as plt
df=pickle.load(open('dft_energies.p','rb'))

plt.plot([0.25,0.50],df['PBE0']['NS']['CHK'][1:]-df['PBE0']['NS']['FLP1'][1:],label='CHK',marker='o',c='b')
plt.plot([0.25,0.50],df['PBE0']['NS']['COL'][1:]-df['PBE0']['NS']['FLP1'][1:],label='COL',marker='o',c='g')
plt.plot([0.25,0.50],df['PBE0']['NS']['FLP1'][1:]-df['PBE0']['NS']['FLP1'][1:],label='FLP1',marker='s',c='r')
plt.plot([0.25,0.50],df['PBE0']['NS']['FLP3'][1:]-df['PBE0']['NS']['FLP1'][1:],label='FLP3',marker='s',c='k')
plt.plot([0.25,0.50],df['PBE0']['NS']['FM'][1:]-df['PBE0']['NS']['FLP1'][1:],label='FM',marker='^',c='b')
plt.ylabel("E-E0, Ha")
plt.legend(loc=1)
plt.show()
