import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam, Adagrad, RMSprop
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.layers import Input, Dense
from keras.models import Model

K.clear_session()
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

inputs = Input(shape=(13,))
x = Dense(8, activation='tanh')(inputs)
x = Dense(5, activation='tanh')(x)
x = Dense(2, activation='tanh')(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(RMSprop(lr=0.05),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y_cat, validation_split=0.2, epochs=200, verbose=0)

inp = model.layers[0].input
out = model.layers[2].output
feature_function = K.function([inp], [out])
features = feature_function([X])[0]
plt.scatter(features[:, 0], features[:, 1], c=y)