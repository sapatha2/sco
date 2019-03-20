import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.decomposition import PCA

df=None
df=pd.read_pickle('small_gosling.pickle')
print(df.shape)
#df=df.iloc[[0,15,16,17]]
#df=df.iloc[[0,1,13,26,50]]
#print(df)
#print(df.var())
#exit(0)
#PCA
'''
pca=PCA()
pca.fit(df[['sigT','sigU','sigNpp','sigNps']])
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
print(pca.components_)
exit(0)
'''

'''
ind=list(np.arange(42))+list(np.arange(50,63))
'''
#ind=list(np.arange(7))+list(np.arange(13,13+7))+list(np.arange(26,26+10))+list(np.arange(50,50+8))
#df=df.iloc[26:50]
#df.iloc[3:5]['Sz']=12
'''
for state in ['chk','col','flp']:
  d=pd.read_pickle(state+'_gosling.pickle')
  if(df is None): df=d
  else: df=pd.concat((df,d),axis=0)
df=df.iloc[np.arange(7+7+6)]
'''
'''
df['N']=df['sigNpp']+df['sigNd']+df['sigNps']
sns.pairplot(df,vars=['energy','sigNpp','sigNd','sigNps','N'],hue='Sz',markers='o')
plt.show()
'''

#print(df[['energy','sigNdz','sigNdpi','sigNpz','sigNdz2','sigN2s']])
#print(df[['energy','sigN4s','sigN2s']])
#print(df[['energy','sigU','sigTd','sigT','sigNpp','sigNps']])
#exit(0)

#Number
#sns.pairplot(df,vars=['energy','sigNdz','sigNdpi','sigNpz','sigNdz2','sigN2s'],hue='Sz',markers='o')
#plt.show()
#plt.close()
#sns.pairplot(df,vars=['energy','sigN4s','sigN2s'],hue='Sz',markers='o')
#plt.show()
#plt.close()
#Hopping + 2body
#sns.pairplot(df,vars=['energy','sigU','sigmo0','sigmo1'],hue='Sz',markers='o')
#plt.show()
#exit()
#sns.pairplot(df,vars=['energy','sigU','sigT','sigNps','sigNd'],hue='Sz',markers='o')
#plt.show()

#df['N']=df['sigNd']+df['sigNps']+df['sigNpp']
#df['n']=df['n_1']-df['n_0']
#print(df)
#sns.pairplot(df,vars=['energy','n'])
#plt.show()

#df['n']=[0.958333,1.958333,1.116858,1.164871,0.952557]
df=df.iloc[np.arange(df.shape[0]-3)]
y=df['energy']
#print(df['n'])
#exit(0)
#X=df[['sigmo0']]
#X=df[['sigT']]
#X=df[['sigU']]
#X=df[['sigNpp']]
#X=df[['sigNps']]
#X=df[['sigT','sigU']]
#X=df[['sigT','sigU','sigNpp']]
X=df[['sigT','sigU','sigNps']]  #Do t and U have to be divided by stuff becasue of the super cell?
#X=df[['sigT','sigU','sigNpp','sigNps']]
X=sm.add_constant(X)
ols=sm.OLS(y,X).fit()
print(ols.summary())
__,l_ols,u_ols=wls_prediction_std(ols,alpha=0.10)

plt.plot(ols.predict(),y,'bo')
for i in range(len(y)):
  plt.plot([l_ols.values[i],u_ols.values[i]],[y.values[i],y.values[i]],'b-')
plt.plot(y,y,'g--')
plt.show()
