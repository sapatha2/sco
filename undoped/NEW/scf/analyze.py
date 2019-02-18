import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels.api as sm

df=pd.read_pickle('scf_gosling.pickle')

'''
df['N']=df['sigNpp']+df['sigNd']+df['sigNps']
sns.pairplot(df,vars=['energy','sigNpp','sigNd','sigNps','N'],hue='Sz',markers='o')
plt.show()
'''

#sns.pairplot(df,vars=['energy','sigNpp','sigU','sigT'],hue='Sz',markers='o')
#plt.show()

#Number
#sns.pairplot(df,vars=['energy','sigNdz','sigNdpi','sigNpz','sigNdz2','sigN2s'],hue='Sz',markers='o')
#plt.show()
#plt.close()
sns.pairplot(df,vars=['energy','sigN4s','sigN2s'],hue='Sz',markers='o')
plt.show()
plt.close()
#Hopping + 2body
sns.pairplot(df,vars=['energy','sigU','sigTd','sigT','sigNd','sigNpp','sigNps'],hue='Sz',markers='o')
plt.show()
exit()

'''
y=df['energy']
X=df[['sigTd','sigU']]
X=sm.add_constant(X)
ols=sm.OLS(y,X).fit()
plt.plot(ols.predict(),y,'o')
plt.plot(y,y,'--')
plt.show()
'''
