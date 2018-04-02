import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def pretty_confusion_matrix(y_true, y_pred, labels=["False", "True"]):
    cm = confusion_matrix(y_true, y_pred)
    pred_labels = ['Predicted '+ l for l in labels]
    df = pd.DataFrame(cm, index=labels, columns=pred_labels)
    return df

df = pd.read_csv('../data/diabetes.csv')
plt.figure(figsize=(15, 15))

for i, feature in enumerate(df.columns):
    plt.subplot(3, 3, i+1)
    df[feature].plot(kind='hist')
    plt.xlabel(feature)
    
sns.pairplot(df, hue="Outcome")

ss = StandardScaler()
mms = MinMaxScaler()
y = df['Outcome']

# Only using the features that have high correlation
X = pd.DataFrame(data = df, columns = ["Glucose","Insulin","BMI","Age"])

for col in X.columns:
    X[col] = ss.fit_transform(X[[col]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
model = Sequential()
model.add(Dense(8, input_dim=4, activation='tanh'))
model.add(Dense(4, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(Adam(lr=0.05),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, validation_split=0.1, epochs=300, verbose=0)

y_train_pred = model.predict_classes(X_train)
y_test_pred = model.predict_classes(X_test)


acc = accuracy_score(y_train, y_train_pred)
print("Accuracy score (Train set):\t{:0.3f}".format(acc))

acc = accuracy_score(y_test, y_test_pred)
print("Accuracy score (Test set):\t{:0.3f}".format(acc))

print(classification_report(y_test, y_test_pred))
pretty_confusion_matrix(y_test, y_test_pred, ['Not Diabetes', 'Diabetes'])