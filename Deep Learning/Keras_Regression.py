# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Supress Warnings
import warnings
warnings.filterwarnings("ignore")

# Acquiring Data
X = pd.DataFrame(fetch_california_housing().data, columns=fetch_california_housing().feature_names)
y = pd.DataFrame(fetch_california_housing().target, columns=["y"])

# Train-Test-Validation Split
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Declaring the layers of the model
model = keras.Sequential(
    [
     keras.Input(shape=X_train.shape[1:]),
     layers.Dense(40,activation='relu'),
     layers.Dense(20,activation='relu'),
     layers.Dense(10,activation='relu'),
     layers.Dense(1,activation='linear')
      ]
    )

# Model Training
model.compile(loss="mse", optimizer="SGD")
training  = model.fit(X_train,y_train,epochs=50,validation_data=(X_valid, y_valid))
training_df = pd.DataFrame(training.history)
plt.plot(training_df)
plt.grid()

# Evaluation & Prediction
mse_test = model.evaluate(X_test, y_test) # 0.279 MSE
y_pred = model.predict(X_test[:5])


                                         ## CALLBACKS ##
                                         
# Model saving
model_callback_all = keras.callbacks.ModelCheckpoint('saved_models/Keras_Regression', save_best_only=False)
model_callback = keras.callbacks.ModelCheckpoint('saved_models/Keras_Regression/best_model.h5', save_best_only=True)

# Early stopping
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True) # Stopped at 96 epochs

# Learning Rate Scheduler
def lrn_rt_sch(epoch,lr):
    if epoch<5:
        return lr
    else:
        return lr*0.99
    
learning_rt_sch = keras.callbacks.LearningRateScheduler(lrn_rt_sch,verbose=1)

# Re-training
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-3))
training2 = model.fit(X_train, y_train, epochs=100,validation_data=(X_valid, y_valid), \
                    callbacks=[model_callback, early_stopping_cb,learning_rt_sch])
print(model.evaluate(X_test, y_test))  #0.3646

# Custom Callback
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # keys = list(logs.keys())
        print("End epoch {} of training loss :{:.4f} ; val loss: {:.4f} Through custom callback \
              ".format(epoch, logs['loss'],logs['val_loss']))
        
# Re-training
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-3))
training3 = model.fit(X_train, y_train, epochs=10,validation_data=(X_valid, y_valid), \
                    callbacks=[CustomCallback()])
    