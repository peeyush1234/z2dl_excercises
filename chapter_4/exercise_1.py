#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv('./data/diabetes.csv')
plt.figure(figsize=(15, 15))

for i, feature in enumerate(df.columns):
    plt.subplot(3, 3, i+1)
    df[feature].plot(kind='hist')
    plt.xlabel(feature)
    
sns.pairplot(df, hue="Outcome")

ss = StandardScaler()
mms = MinMaxScaler()
y = df['Outcome']
X = df.drop('Outcome', 1)
for col in X.columns:
    X[col] = mms.fit_transform(X[[col]])