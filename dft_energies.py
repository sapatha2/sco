import pickle
import numpy as np
import pandas as pd
data={
  #Ordering: x=0, x=0.25_fv, x=0.25
  'PBE0':{
    'S':{
      'x':['0','0.25_fv','0.25'],
      'CHK':np.array([-9.2123749620223E+02,-9.2092111151634E+02,-9.2092791881441E+02])*27.2,
      'COL':np.array([-9.2122638412999E+02,-9.2091969201656E+02,-9.2092670972882E+02])*27.2,
      'FLP1':np.array([-9.2122636552756E+02,-9.2092669006119E+02,-9.2093343498725E+02])*27.2,
      'FLP3':np.array([-9.2122636552756E+02,-9.2091219092014E+02,-9.2091978045593E+02])*27.2,
      'FM':np.array([-9.2121137910381E+02,-9.2091264058011E+02,-9.2092026703934E+02])
    },
    'NS':{
      'x':['0','0.25_fv','0.25'],
      'CHK':np.array([-9.2123749620223E+02,-9.2092669002320E+02,-9.2093343507729E+02])*27.2,
      'COL':np.array([-9.2122638412999E+02,-9.2092063265454E+02,-9.2092758720044E+02])*27.2,
      'FLP1':np.array([-9.2122636552756E+02,-9.2092669022287E+02,-9.2093343503482E+02])*27.2,
      'FLP3':np.array([-9.2122636552756E+02,-9.2091219109429E+02,-9.2091978048822E+02])*27.2,
      'FM':np.array([-9.2121137910381E+02,-9.2091264054414E+02,-9.2092002873599E+02])*27.2
    },
  },
  'HSE06':{
    'S':{
      'x':['0','0.25_fv','0.25'],
      'CHK':np.array([-9.2122881518088E+02,0,0])*27.2,
      'COL':np.array([-9.2121750188336E+02,0,0])*27.2,
      'FLP1':np.array([-9.2121759082240E+02,0,0])*27.2,
      'FLP3':np.array([-9.2121759082240E+02,0,0])*27.2,
      'FM':np.array([-9.2120231036158E+02,0,0])*27.2
    },
    'NS':{
      'x':['0','0.25_fv','0.25'],
      'CHK':np.array([-9.2122881518088E+02,-9.2092109901659E+02,0])*27.2,
      'COL':np.array([-9.2121750188336E+02,-9.2091332695333E+02,0])*27.2,
      'FLP1':np.array([-9.2121759082240E+02,-9.2092109920075E+02,0])*27.2,
      'FLP3':np.array([-9.2121759082240E+02,-9.2090821996870E+02,0])*27.2,
      'FM':np.array([-9.2120231036158E+02,-9.2090837968879E+02,0])*27.2
    },
  }
}
  
df=pd.DataFrame(data=data)
pickle.dump(df,open('dft_energies.p','wb'))

import matplotlib.pyplot as plt
df=pickle.load(open('dft_energies.p','rb'))
#Undoped
plt.plot([-0.01],df['PBE0']['NS']['CHK'][0]-df['PBE0']['NS']['CHK'][0],label='CHK -> FLP1',marker='o',c='b')
plt.plot([-0.01],df['PBE0']['NS']['COL'][0]-df['PBE0']['NS']['CHK'][0],label='COL -> ?',marker='o',c='g')
plt.plot([-0.01],df['PBE0']['NS']['FLP1'][0]-df['PBE0']['NS']['CHK'][0],label='FLP -> FLP1',marker='s',c='r')
plt.plot([-0.01],df['PBE0']['NS']['FLP3'][0]-df['PBE0']['NS']['CHK'][0],label='FLP -> ?',marker='s',c='k')
plt.plot([-0.01],df['PBE0']['NS']['FM'][0]-df['PBE0']['NS']['CHK'][0],label='FM -> FM3',marker='^',c='b')

plt.plot([0.01],df['HSE06']['NS']['CHK'][0]-df['HSE06']['NS']['CHK'][0],marker='o',c='b')
plt.plot([0.01],df['HSE06']['NS']['COL'][0]-df['HSE06']['NS']['CHK'][0],marker='o',c='g')
plt.plot([0.01],df['HSE06']['NS']['FLP1'][0]-df['HSE06']['NS']['CHK'][0],marker='s',c='r')
plt.plot([0.01],df['HSE06']['NS']['FLP3'][0]-df['HSE06']['NS']['CHK'][0],marker='s',c='k')
plt.plot([0.01],df['HSE06']['NS']['FM'][0]-df['HSE06']['NS']['CHK'][0],marker='^',c='b')

#Doped
plt.plot([0.24],df['PBE0']['NS']['CHK'][1]-df['PBE0']['NS']['FLP1'][1],marker='o',c='b')
plt.plot([0.24],df['PBE0']['NS']['COL'][1]-df['PBE0']['NS']['FLP1'][1],marker='o',c='g')
plt.plot([0.24],df['PBE0']['NS']['FLP1'][1]-df['PBE0']['NS']['FLP1'][1],marker='s',c='r')
plt.plot([0.24],df['PBE0']['NS']['FLP3'][1]-df['PBE0']['NS']['FLP1'][1],marker='s',c='k')
plt.plot([0.24],df['PBE0']['NS']['FM'][1]-df['PBE0']['NS']['FLP1'][1],marker='^',c='b')

plt.plot([0.26],df['HSE06']['NS']['CHK'][1]-df['HSE06']['NS']['FLP1'][1],marker='o',c='b')
plt.plot([0.26],df['HSE06']['NS']['COL'][1]-df['HSE06']['NS']['FLP1'][1],marker='o',c='g')
plt.plot([0.26],df['HSE06']['NS']['FLP1'][1]-df['HSE06']['NS']['FLP1'][1],marker='s',c='r')
plt.plot([0.26],df['HSE06']['NS']['FLP3'][1]-df['HSE06']['NS']['FLP1'][1],marker='s',c='k')
plt.plot([0.26],df['HSE06']['NS']['FM'][1]-df['HSE06']['NS']['FLP1'][1],marker='^',c='b')

plt.title("DFT Energies, PBE0 and HSE06")
plt.ylabel(r"$E-E_0$, eV")
plt.xlabel("x")
plt.legend(loc=1)
plt.show()
