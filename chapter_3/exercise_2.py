%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv('../data/HR_comma_sep.csv')

print(df.columns)

ss = StandardScaler()
#mms = MinMaxScaler()
for col in ['number_project',
            'average_montly_hours',
            'time_spend_company',
           ]:
    df[col] = mms.fit_transform(df[[col]])

df = pd.concat([df.drop('salary', axis=1), pd.get_dummies(df['salary'], prefix='salary')], axis=1)
df = pd.concat([df.drop('sales', axis=1), pd.get_dummies(df['sales'], prefix='dep')], axis=1)

model = Sequential()
model.add(Dense(1, input_dim=20))
model.add(Activation('sigmoid'))
model.compile(SGD(lr=0.8), 'binary_crossentropy', metrics=['accuracy'])
y = df['left']
X = df.drop('left', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2)
model.fit(X_train, y_train, epochs=10, verbose=1)
acc = accuracy_score(y_train, model.predict(X_train) > 0.5)
print("Train accuracy score {:0.3f}".format(acc))
acc = accuracy_score(y_test, model.predict(X_test) > 0.5)
print("Test accuracy score {:0.3f}".format(acc))

def pretty_confusion_matrix(y_true, y_pred, labels=["False", "True"]):
    cm = confusion_matrix(y_true, y_pred)
    pred_labels = ['Predicted '+ l for l in labels]
    df = pd.DataFrame(cm, index=labels, columns=pred_labels)
    return df

pretty_confusion_matrix(y, model.predict(X)>0.5, ['Not Left', 'Left'])
print(classification_report(y, model.predict(X)>0.5))
