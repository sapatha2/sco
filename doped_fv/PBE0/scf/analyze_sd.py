import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

df=pd.read_pickle('pickles/full_sd_gosling.pickle')

#TRACE CHECK
df['tr']=df['Nd']+df['Ns']+df['Np']#+df['Nsr']#+df['Ndz2']+df['Ndz']+df['Ndpi']+df['Npz']+df['Npp']+df['N2s']
df['tr_group']=0
df['tr_group'][df['tr']<=20.7]=-1
df['tr_group'][df['tr']>=21]=1

df['tr_unp']=df['N0']+df['N1']+df['N2']+df['N3']+df['N4']+df['N5']+df['N6']+df['N7']

#COMPOSITE VARS
df['N1']+=df['N2']
df['N5']+=df['N6']
df['sigT_0_2']+=df['sigT_0_3']
df['sigT_0_5']+=df['sigT_0_6']
df['sigT_1_2']+=df['sigT_1_3']
df['sigT_1_5']+=df['sigT_1_6']
df['sigT_2_4']+=df['sigT_3_4']
df['sigT_2_5']+=df['sigT_3_5']+df['sigT_2_6']+df['sigT_3_6']
df['sigT_2_7']+=df['sigT_3_7']
df['sigT_4_5']+=df['sigT_4_6']
df['sigT_5_7']+=df['sigT_6_7']
df=df.drop(columns=['N2','N6','sigT_0_3','sigT_0_6',
'sigT_1_3','sigT_1_6','sigT_3_4','sigT_3_5','sigT_2_6',
'sigT_3_6','sigT_3_7','sigT_4_6','sigT_6_7','sigT_2_3','sigT_5_6'])

#VARIANCES
var=df.var()
ind=np.argsort(var.values)
print(var.iloc[ind])
#exit(0)

#PCA
var=['Nd','Np','Ns','Nsr','sigTdp','sigTps','sigTds','sigUd','sigUp','sigUs']
X=df[var]
X=sm.add_constant(X)
pca=PCA(n_components=len(list(X)))
pca.fit(X)
print(list(X))
for i in range(pca.components_.shape[0]):
  print(np.around(pca.explained_variance_ratio_[i],3),
  np.around(pca.components_[i,:],3)) #(component, feature)

var=['N1','N3','N4','N5','N7','sigUd','sigUp','sigUs']
X=df[var]
X=sm.add_constant(X)
pca=PCA(n_components=len(list(X)))
pca.fit(X)
print(list(X))
for i in range(pca.components_.shape[0]):
  print(np.around(pca.explained_variance_ratio_[i],3),
  np.around(pca.components_[i,:],3)) #(component, feature)
#exit(0)

#CORR MATRIX
#var=['energy','Nd','Np','Ns','Nsr','sigTdp','sigTps','sigTds','sigUd','sigUp','sigUs']
#var=['energy','Nd','Np','Ns','Nsr','sigTdp','sigTps','sigTds']
#plt.matshow(df[var].corr(),vmin=-1,vmax=1,cmap=plt.cm.bwr)
#plt.show()
#exit(0)

#PAIRPLOTS
#sns.pairplot(df,vars=['energy','Np','sigTdp','sigTps'],hue='basestate')
#plt.show()
#exit(0)
#sns.pairplot(df,vars=['energy','tr','tr_unp'],hue='tr_group')
#plt.show()
'''
sns.pairplot(df,vars=['energy','Nd','Np','Ns','Nsr','sigTdp','sigTps','sigTds','sigUd','sigUp','sigUs'],hue='basestate')
plt.savefig('HubbardPPIAO_full.pdf',bbox_inches='tight')
plt.close()
sns.pairplot(df,vars=['energy','N0','N1','N3','N4','N5','N7','sigUd','sigUp','sigUs'],hue='basestate')
plt.savefig('HubbardPPMO_full.pdf',bbox_inches='tight')
plt.close()
exit(0)
'''

#CORRELATION SCATTER
'''
df=sm.add_constant(df)
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['sigTdp'], df['sigTps'], df['Np'])
ax.set_xlabel('sigTdp')
ax.set_ylabel('sigTps')
ax.set_zlabel('Np')
plt.show()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['sigTdp'], df['sigTps'], df['sigUd'])
ax.set_xlabel('sigTdp')
ax.set_ylabel('sigTps')
ax.set_zlabel('sigUd')
plt.show()
'''
#exit(0)

#REGRESSION
select_df=sm.add_constant(df)
y=select_df['energy']

#MULTICOLLINEARITY CHECKS 
#y=select_df['Np']
#var=['Ns','Np','Nsr','sigTdp','sigTps','sigTds','sigUd','sigUs']
#y=select_df['sigUs']
#var=['N1','N3','N4','N5','N7','sigUd','sigUp','sigUd']

#IAO REGRESSION 
#var=['Np','Ns','Nsr','sigTdp','sigTds','sigTps','sigUd','sigUp','sigUs']
#var=['Np','Ns','Nsr','sigTdp','sigTds','sigTps','sigUd']

#MO REGRESSION 
var=['N1','N3','N4','N5','N7','sigUd','sigUs']#'sigUp','sigUs']
#var=['N1','N3','N4','N5','N7','sigUd','sigUp','sigUs']
#for i in list(select_df):
#  if('sigT_' in i): var+=[i]

X=select_df[var]  #Do t and U have to be divided by stuff because of the super cell?
y=select_df['energy']
X=sm.add_constant(X)

beta=0.0
weights=np.exp(-beta*(y-min(y)))
ols=sm.WLS(y,X,weights).fit()
print(ols.summary())

#ADD NOISE 
for mag in [0.10]:
  for run in range(50):
    #X=select_df[var]  #Do t and U have to be divided by stuff because of the super cell?
    #X+=np.random.normal(scale=mag,size=X.shape)
    y=select_df['energy']
    y+=np.random.normal(scale=mag,size=y.shape)
    X=sm.add_constant(X)
    
    beta=0.0
    weights=np.exp(-beta*(y-min(y)))
    ols=sm.WLS(y,X,weights).fit()
 
    zz=(ols.conf_int()[0]-ols.conf_int()[1])/2
    plt.errorbar(np.arange(len(ols.params)-1),ols.params[1:],zz.values[1:],marker='.',c='b',ls='None')
plt.xticks(np.arange(len(list(X))-1),list(X)[1:])
plt.show()
exit(0)

__,l_ols,u_ols=wls_prediction_std(ols,alpha=0.10)
select_df['energy_err']=0
select_df['pred']=ols.predict(X)
select_df['pred_err']=(u_ols-l_ols)/2
select_df['resid']=select_df['energy']-select_df['pred']

g = sns.FacetGrid(select_df,hue='basestate')
g.map(plt.errorbar, "pred", "energy", "energy_err",fmt='o').add_legend()
plt.plot(select_df['energy'],select_df['energy'],'k--')
plt.show()
exit(0)
