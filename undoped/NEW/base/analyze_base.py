import os 
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

def analyze(df):
  print(df)
  #sns.pairplot(df,vars=['energy','sigT','sigU','sigJ'],hue='state')
  #plt.show()
  #plt.savefig('plots/base_vmc.pdf',bbox_inches='tight')

  df['calc']='vmc' #VMC
  df=df.drop(columns=['energy','energy_err'])
  df_scf=np.load('scf/base_scf.pickle')
  df_scf['calc']='scf' #SCF
  df=pd.concat((df,df_scf),axis=0)
  print(df)
  sns.pairplot(df,vars=['sigT','sigU','sigJ'],hue='calc',markers=['o','s'])
  #plt.show()
  plt.savefig('plots/base_comp.pdf',bbox_inches='tight')

if __name__=='__main__':
  df=np.load('base_gosling.pickle')
  analyze(df)
