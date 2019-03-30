import os 
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm
def analyze(df):
  #Pairplot
  #print(df.shape)
  boolean=(df['path']=='1')
  for p in [1,2,3,4]:
    boolean+=df['path']==str(p)
  df=df[boolean]
  #print(list(df))
  '''
  sns.pairplot(df,vars=['energy','sigT','sigTd','sigU','sigNpp','sigNps','sigN4s'],hue='path',palette=sns.color_palette("husl", 9))
  plt.show()
  plt.close()
  exit(0)
  sns.pairplot(df,vars=['energy','sigNd','sigNpp','sigNps','sigN4s'],hue='path',palette=sns.color_palette("husl", 4))
  plt.show()
  plt.close()
  '''
  sns.pairplot(df,vars=['energy','sigT','sigU','sigNpp','sigNps'],hue='path',palette=sns.color_palette("husl", 9))
  plt.show()
  #plt.savefig('plots/vmc_pairplot.pdf',bbox_inches='tight')
  exit(0)

  #Fit
  y=df['energy']
  X=df[['sigT','sigU','sigNpp','sigNps']]
  #X=df[['sigTd','sigU','sigNpp']]
  X=sm.add_constant(X)
  beta=0.5
  ols=sm.WLS(y,X,weights=np.exp(-beta*(y-min(y)))).fit()
  print(ols.summary())
  df['pred']=ols.predict(X)
  #df['resid']=df['energy']-df['pred']
  #sns.pairplot(df,vars=['energy','pred'],hue='path',palette=sns.color_palette("husl", 9))
  #sns.regplot(x='pred',y='energy',data=df)
  plt.errorbar(df['pred'],df['energy'],yerr=df['energy_err'],fmt='o')
  plt.errorbar(df['energy'],df['energy'],yerr=df['energy_err'],fmt='--')
  plt.show()
  #exit(0)
  
  '''
  for p in np.arange(1,15):
    d=df[df['path']==str(p)]
    plt.errorbar(d['pred'],d['energy'],yerr=d['energy_err'],fmt='o',label='path '+str(p))
  plt.plot(df['energy'],df['energy'],'g-')
  plt.ylabel('energy (eV)')
  plt.xlabel('pred (eV)')
  plt.legend(loc='best')
  #plt.savefig('plots/vmc_pred_Td.pdf',bbox_inches='tight')
  plt.show()
  exit(0)
  '''
 
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
  indSz0=18

  ax.plot([fx[0]], [fy[0]], [fz[0]], "bo",label='Sz=0')
  ax.plot([fx[indSz0]], [fy[indSz0]], [fz[indSz0]], "go",label='Sz=2')
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
  '''
if __name__=='__main__':
  df=np.load('b_gosling.pickle')
  analyze(df)
