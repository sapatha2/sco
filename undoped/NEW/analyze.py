import os 
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm

def analyze(df):
  print(df)

  '''
  #Residuals
  y=df['energy']
  X=df[['sigU']]
  X=sm.add_constant(X)
  ols=sm.OLS(y,X).fit()
  print(ols.summary())
  df['resid']=df['energy']-ols.predict(X)
  sns.pairplot(df,vars=['resid','sigT','sigU','sigJ'],hue='path',markers=['s']*3+['o']*3)
  plt.show()
  '''

  #Pairplot
  #df['sigNd+sigNps']=df['sigNd']+df['sigNps']
  #sns.pairplot(df,vars=['energy','sigTd','sigU'],hue='path',markers=['s']*3+['o']*3)
  #plt.savefig('plots/vmc_pairplot.pdf',bbox_inches='tight')
  #plt.close()
  #plt.show()

  #Fit
  y=df['energy']
  X=df[['sigTd','sigU']]
  X=sm.add_constant(X)
  ols=sm.OLS(y,X).fit()
  print(ols.summary())
  '''
  plt.errorbar(ols.predict(X),df['energy'],yerr=df['energy_err'],fmt='bo')
  plt.plot(df['energy'],df['energy'],'g-')
  plt.ylabel('energy (eV)')
  plt.xlabel('pred (eV)')
  plt.savefig('plots/vmc_pred_Td.pdf',bbox_inches='tight')
  '''

  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  #Values with errorbars
  f=df.values
  fx=f[:,3] #sigTd
  fy=f[:,4] #sigU
  fz=f[:,0] #energy
  zerror=f[:,1] #energy_err
  ax.plot(fx, fy, fz, "bo")
  for i in np.arange(0, len(fx)):
    ax.plot([fx[i], fx[i]], [fy[i], fy[i]], [fz[i]+zerror[i], fz[i]-zerror[i]], "b_")
  
  #Plane!
  gridx=np.linspace(min(fx),max(fx),100)
  gridy=np.linspace(min(fy),max(fy),100)
  xv,yv=np.meshgrid(gridx,gridy)
  zv=ols.params[0] + ols.params[1]*xv + ols.params[2]*yv 
  ax.plot_surface(xv,yv,zv,cmap=plt.cm.coolwarm)

  ax.set_xlabel('sigTd')
  ax.set_ylabel('sigU')
  ax.set_zlabel('E (eV)')
  plt.show()

if __name__=='__main__':
  df=np.load('p_gosling.pickle')
  analyze(df)
