import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("../data/wines.csv")
target_names = df['Class'].unique()
target_dict = {n:i for i, n in enumerate(target_names)}

y = df['Class'].map(target_dict)
y_cat = to_categorical(y)
X = df.drop('Class', axis=1)
#sns.pairplot(df, hue="Class")

ss = StandardScaler()
mms = MinMaxScaler()
for col in X.columns:
    X[col] = ss.fit_transform(X[[col]])

model = Sequential()
model.add(Dense(8, input_dim=13, activation='tanh'))
model.add(Dense(4, activation='tanh'))
model.add(Dense(3, activation='softmax'))
model.compile(Adam(lr=0.05),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y_cat, validation_split=0.2, epochs=400, verbose=1, batch_size=64)