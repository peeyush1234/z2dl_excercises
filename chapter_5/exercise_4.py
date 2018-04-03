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
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import keras.backend as K

K.clear_session()
df = pd.read_csv("../data/wines.csv")
target_names = df['Class'].unique()
target_dict = {n:i for i, n in enumerate(target_names)}

y = df['Class'].map(target_dict)
y_cat = to_categorical(y)
X = df.drop('Class', axis=1)
sns.pairplot(df, hue="Class")

ss = StandardScaler()
mms = MinMaxScaler()
for col in X.columns:
    X[col] = ss.fit_transform(X[[col]])

    
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3, random_state=42)
inputs = Input(shape=(13,))
x = Dense(8, activation='tanh')(inputs)
x = Dense(5, activation='tanh')(x)
x = Dense(2, activation='tanh')(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(RMSprop(lr=0.05),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

earlystop = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=20, \
                          verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath='../data/tmp/model', verbose=1, save_best_only=True)
tensor_board = TensorBoard(log_dir='../data/tmp/logs')
callbacks_list = [earlystop, checkpointer, tensor_board]

model.fit(X_train, y_train, callbacks=callbacks_list, validation_data=(X_test, y_test), epochs=300, verbose=1)
