import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np

df = pd.read_csv('Churn.csv') #daten werden eingelesen!?

data = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1)) #daten werden definiert, wobei die customer id und churn entfernt werden und der rest der daten mit pd.get_dummies in binäre form gebracht werden
labels = df['Churn'].apply(lambda x: 1 if x=='Yes' else 0) #labels werden extrahiert und mit agrument und funktion lambda zu O oder 1 umgewandelt



train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=.2) # test size legt proportion von trainings zu testdatensatz fest, hier 0.2

# y_train.head()
print(train_data)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=len(train_data.columns)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')

train_data = np.asarray(train_data).astype('float32')
train_labels = np.asarray(train_labels).astype('float32')

model.fit(train_data, train_labels, epochs=2, batch_size=32) #original e= 200 bs= 32

test_data = np.asarray(test_data).astype('float32')
test_labels = np.asarray(test_labels).astype('float32')

model.evaluate(test_data, test_labels)



#y_hat = model.predict(X_test)
#y_hat = [0 if val < 0.5 else 1 for val in y_hat]


#print(accuracy_score(y_test, y_hat))

