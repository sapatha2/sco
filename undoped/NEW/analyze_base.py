import os 
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

def analyze(df):
  print(df)
  sns.pairplot(df,vars=['energy','sigT','sigU','sigJ'],hue='state')
  #plt.show()
  plt.savefig('plots/base_vmc.pdf',bbox_inches='tight')

if __name__=='__main__':
  df=np.load('base_gosling.pickle')
  analyze(df)
