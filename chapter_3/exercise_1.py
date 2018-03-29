%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('../data/housing-data.csv')

plt.figure(figsize=(15, 5))

for i, feature in enumerate(df.columns):
    plt.subplot(1, 4, i+1)
    df[feature].plot(kind='hist', title=feature)
    plt.xlabel(feature)


ss = StandardScaler()
#mms = MinMaxScaler()
for col in df.columns:
    df[col] = ss.fit_transform(df[[col]])

y = df['price']
X = df[['sqft', 'bdrms', 'age']]

model = Sequential()
model.add(Dense(1, input_shape=(3,)))
model.compile(SGD(lr=0.1), 'mean_squared_error')
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2)
model.fit(X_train, y_train, epochs=500, verbose=1)
y_train_pred = model.predict(X_train).ravel()
y_test_pred = model.predict(X_test).ravel()

r2 = r2_score(y_train, y_train_pred)
print("R2 score (Train set):\t{:0.3f}".format(r2))

r2 = r2_score(y_test, y_test_pred)
print("R2 score (Test set):\t{:0.3f}".format(r2))
