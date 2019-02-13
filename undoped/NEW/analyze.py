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
  df['pred']=ols.predict(X)
  for p in np.arange(1,7):
    d=df[df['path']==str(p)]
    plt.errorbar(d['pred'],d['energy'],yerr=d['energy_err'],fmt='o',label='path '+str(p))
  plt.plot(df['energy'],df['energy'],'g-')
  plt.ylabel('energy (eV)')
  plt.xlabel('pred (eV)')
  plt.legend(loc='best')
  plt.savefig('plots/vmc_pred_Td.pdf',bbox_inches='tight')
  exit(0)

  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  #Values with errorbars
  f=df.values
  fx=f[:,3] #sigTd
  fy=f[:,4] #sigU
  fz=f[:,0] #energy
  zerror=f[:,1] #energy_err
  indSz0=18

  ax.plot([fx[0]], [fy[0]], [fz[0]], "bo",label='Sz=0')
  ax.plot([fx[indSz0]], [fy[indSz0]], [fz[indSz0]], "go",label='Sz=1')
  for i in np.arange(indSz0):
    ax.plot([fx[i]], [fy[i]], [fz[i]], "bo")
    ax.plot([fx[i], fx[i]], [fy[i], fy[i]], [fz[i]+zerror[i], fz[i]-zerror[i]], "b_")
  for i in np.arange(indSz0,len(fx)):
    ax.plot([fx[i]], [fy[i]], [fz[i]], "go")
    ax.plot([fx[i], fx[i]], [fy[i], fy[i]], [fz[i]+zerror[i], fz[i]-zerror[i]], "g_")

  #Plane!
  gridx=np.linspace(min(fx),max(fx),100)
  gridy=np.linspace(min(fy),max(fy),100)
  xv,yv=np.meshgrid(gridx,gridy)
  zv=ols.params[0] + ols.params[1]*xv + ols.params[2]*yv 
  ax.plot_surface(xv,yv,zv,cmap=plt.cm.coolwarm)

  ax.set_xlabel('sigTd')
  ax.set_ylabel('sigU')
  ax.set_zlabel('E (eV)')
  plt.legend(loc='best')
  plt.show()

if __name__=='__main__':
  df=np.load('p_gosling.pickle')
  analyze(df)
