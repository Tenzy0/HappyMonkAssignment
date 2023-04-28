import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Input
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
batch_size=569
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()
dataset.values()
df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
X = np.array(df)
y = dataset.target
X.shape, y.shape
input_layer = Input(shape=(None, 30))
layer_1 = Dense(512, activation = 'relu')(input_layer)
output_layer = Dense(1, activation = 'sigmoid')(layer_1)
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
history = model.fit(X, y, batch_size=112, epochs=10)#,validation_data=(X_test, y_test)) 

