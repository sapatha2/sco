import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels.api as sm

df=None
df=pd.read_pickle('scf_gosling.pickle')
df.iloc[3:5]['Sz']=12
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

#Number
#sns.pairplot(df,vars=['energy','sigNdz','sigNdpi','sigNpz','sigNdz2','sigN2s'],hue='Sz',markers='o')
#plt.show()
#plt.close()
#sns.pairplot(df,vars=['energy','sigN4s','sigN2s'],hue='Sz',markers='o')
#plt.show()
#plt.close()
#Hopping + 2body
#sns.pairplot(df,vars=['energy','sigU','sigTd','sigT','sigNd','sigNpp','sigNps'],hue='Sz',markers='o')
#plt.show()
#exit()
sns.pairplot(df,vars=['energy','sigU','sigTd','sigT','sigNpp','sigNps'],hue='Sz',markers='o')
plt.show()
exit(0)

y=df['energy']
#X=df[['sigT','sigU','sigNpp','sigNps']]
X=df[['sigTd','sigU','sigNpp']]
X=sm.add_constant(X)
ols=sm.OLS(y,X).fit()
print(ols.summary())
plt.plot(ols.predict(),y,'o')
plt.plot(y,y,'--')
plt.show()
