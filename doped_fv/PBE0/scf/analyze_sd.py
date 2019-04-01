import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.decomposition import PCA

df=pd.read_pickle('pickles/sd_gosling.pickle')
print(df)
exit(0)
#df=df.iloc[[0,12,24,36]]
var=df.var()
ind=np.argsort(var.values)
print(var.iloc[ind])

def plot_parm(parm):
  for i in range(12):
    plt.plot(df.iloc[[0,i]][parm],df.iloc[[0,i]]['energy'],'bo-')
  for i in range(12,24):
    plt.plot(df.iloc[[12,i]][parm],df.iloc[[12,i]]['energy'],'go-')
  for i in range(24,36):
    plt.plot(df.iloc[[24,i]][parm],df.iloc[[24,i]]['energy'],'ro-')
  for i in range(36,48):
    plt.plot(df.iloc[[36,i]][parm],df.iloc[[36,i]]['energy'],'ko-')
  plt.show()
  return 
plot_parm('sigNps')
exit(0)

#PAIRPLOTS
#sns.pairplot(df,vars=['energy','sigNdpi','sigNdz','sigNpp','sigNpz'],hue='basestate')
#sns.pairplot(df,vars=['energy','sigN2s','sigNdz2'],hue='basestate')
#sns.pairplot(df,vars=['energy','sigNps','sigNd','sigN4s'],hue='basestate')
#sns.pairplot(df,vars=['energy','sigTd','sigT','sigU','sigJ'],hue='basestate')
#sns.pairplot(df,vars=['energy','sigT','sigTd','sigJ','sigNps'],hue='basestate')
sns.pairplot(df,vars=['energy','sigN4s','sigU','sigNd'],hue='basestate')
plt.show()
exit(0)

'''
#APPLYING THE CUTOFFS:
ind=np.where(df['energy']==min(df['energy']))[0][0]

#Npi, Nz
cutoff=10
select=np.full(df.shape[0],True)
for field in ['sigNpp','sigNdpi','sigNpz','sigNdz']:
  n_gs=df.iloc[ind][field]
  select*=(df[field].values<(n_gs+cutoff))*\
         (df[field].values>(n_gs-cutoff))

#Energy
cutoff=10.0
select*=(df['energy'].values<=min(df['energy'])+cutoff)

#Plots
select_df=df.iloc[select]
sns.pairplot(select_df,vars=['energy','sigTd','sigT','sigNps','sigU'],hue='Sz')
#sns.pairplot(select_df,vars=['energy','n','sigN4s','n_tot'],hue='Sz')
plt.show()
exit(0)
'''

#REGRESSION
select_df=df
y=select_df['energy']
X=select_df[['sigJ','sigT']]  #Do t and U have to be divided by stuff because of the super cell?
#X=sm.add_constant(X)
beta=0
weights=np.exp(-beta*(y-min(y)))
ols=sm.WLS(y,X,weights).fit()
print(ols.summary())
__,l_ols,u_ols=wls_prediction_std(ols,alpha=0.10)

select_df['energy_err']=0
select_df['pred']=ols.predict(X)
select_df['pred_err']=(u_ols-l_ols)/2
select_df['resid']=select_df['energy']-select_df['pred']

g = sns.FacetGrid(select_df,hue='basestate')
g.map(plt.errorbar, "pred", "energy", "energy_err","pred_err",fmt='o').add_legend()
plt.plot(select_df['energy'],select_df['energy'],'k--')
plt.show()
exit(0)

#plt.plot(ols.predict(),y,'bo')
#for i in range(len(y)):
#  plt.plot([l_ols.values[i],u_ols.values[i]],[y.values[i],y.values[i]],'b-')
#plt.plot(y,y,'g--')
#plt.show()

