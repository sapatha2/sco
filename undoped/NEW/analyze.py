import os 
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm

def analyze(df):
  print(df)
  #sns.pairplot(df,vars=['energy','sigT','sigU','sigJ'],hue='path',markers=['s']*3+['o']*3)
  #plt.show()
  #plt.savefig('plots/base_vmc.pdf',bbox_inches='tight')
  plt.errorbar(df['sigJ'],df['energy'],yerr=df['energy_err'],fmt='o')
  plt.show()  

  '''
  df=df[df['path']=='1']
  y=df['energy']
  #X=df[['sigT','sigU']]
  #X=df[['sigT']]
  X=sm.add_constant(X)
  ols=sm.OLS(y,X).fit()
  print(ols.summary())
  '''
if __name__=='__main__':
  df=np.load('p_gosling.pickle')
  analyze(df)
