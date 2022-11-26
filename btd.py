import os
import cv2
import glob
import random
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import io
from tensorflow.python.keras import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Importing Data
df = pd.read_csv('desktop/python/Brain_MRI_Xception/data_mask.csv')
df_train = df.drop(columns = ['patient_id'])
df_train['mask'] = df_train['mask'].apply(lambda x: str(x))

# Train Test split
train, test = train_test_split(df_train, test_size = 0.2,random_state=42)

# Data Augmentation usnig ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./298., validation_split = 0.1)

# Training data
train_generator=datagen.flow_from_dataframe(
dataframe=train,
directory= './',
x_col='image_path',
y_col='mask',
subset="training",
batch_size=16,
shuffle=True,
class_mode="categorical",
target_size=(299,299))

# Validation data
valid_generator=datagen.flow_from_dataframe(
dataframe=train,
directory= './',
x_col='image_path',
y_col='mask',
subset="validation",
batch_size=16,
shuffle=True,
class_mode="categorical",
target_size=(299,299))

# Test Data Generator
test_datagen=ImageDataGenerator(rescale=1./298.)
test_generator=test_datagen.flow_from_dataframe(
dataframe=test,
directory= './',
x_col='image_path',
y_col='mask',
batch_size=16,
shuffle=False,
class_mode='categorical',
target_size=(299,299))

# Import Pre-Trained Xception Model & Freeze its Weights
pre_trained_model =Xception(weights = 'imagenet', include_top = False, input_tensor = Input(shape=(299, 299, 3)))
for layer in pre_trained_model.layers:
  layers.trainable = False

# Subsequent Layers for Fine Tuning
tunedmodel = pre_trained_model.output
tunedmodel = AveragePooling2D(pool_size = (4,4))(tunedmodel)

# Flatten Layers to convert Feature Maps into Vectors
tunedmodel = Flatten(name= 'flatten')(tunedmodel)

# 4 Sets of Dense and Dropout Layers with Relu Activation
tunedmodel = Dense(299, activation = "relu")(tunedmodel)
tunedmodel = Dropout(0.25)(tunedmodel)#

tunedmodel = Dense(299, activation = "relu")(tunedmodel)
tunedmodel = Dropout(0.25)(tunedmodel)

tunedmodel = Dense(299, activation = "relu")(tunedmodel)
tunedmodel = Dropout(0.25)(tunedmodel)

tunedmodel = Dense(299, activation = "relu")(tunedmodel)
tunedmodel = Dropout(0.25)(tunedmodel)

# Final Output Layer with Softmax Activation
tunedmodel = Dense(2, activation = 'softmax')(tunedmodel)

# Combining all the Layers along with pre_trained Model into the new model
model = Model(inputs = pre_trained_model.input, outputs = tunedmodel)

# Define Early Stopping Criteria, Check Pointer, LR decay
earlystopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              verbose=1, 
                              patience=25
                             )
checkpointer = ModelCheckpoint(filepath="/desktop/python/Brain_MRI_Xception/classifier-Xception-weights.hdf5", 
                               verbose=1, 
                               save_best_only=True
                              )
lr_decay = ReduceLROnPlateau(monitor='val_loss',
                              mode='min',
                              verbose=1,
                              patience=10,
                              min_delta=0.0001,
                              factor=0.2
                             )
# Callbacks for training 
callbacks = [checkpointer, earlystopping, lr_decay]

# Compiling the Model using CE Loss, Adam Optimiser
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics= ["accuracy"])

# Trainig for 100 epochs
history = model.fit(train_generator, 
              steps_per_epoch= train_generator.n // train_generator.batch_size, 
              epochs = 100, 
              validation_data= valid_generator, 
              validation_steps= valid_generator.n // valid_generator.batch_size, 
              callbacks = [checkpointer, earlystopping, lr_decay])

# Saving the Model for evaluation
json = model.to_json()
with open("desktop/python/Brain_MRI_Xception/classifierXceptionmodel.json", "w") as file:
    file.write(json)

# Subsequent Model Evaluation
with open('desktop/python/Brain_MRI_Xception/classifierXceptionmodel.json', 'r') as file:
    Saved_Model= file.read()

model = tf.keras.models.model_from_json(Saved_Model)

# Loading Weights
model.load_weights('/desktop/python/Brain_MRI_Xception/classifier-Xception-weights.hdf5')
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics= ["accuracy"])

# Model Prediction on Unseen Data
test_predict = model.predict(test_generator, steps = test_generator.n // 16, verbose =1)

# Output labels on Test Data
predict = []
for i in test_predict:
  predict.append(str(np.argmax(i)))
predict = np.asarray(predict)
print(predict)

# Model Accuracy
accuracy = round(accuracy_score(original, predict),2)
print(accuracy)

# Model Classification Report
report = classification_report(original, predict, labels = [0,1])
print(report)

# Model Confusion Matrix
cm = confusion_matrix(original, predict)
plt.figure(figsize = (7,7))
sns.heatmap(cm, annot=True,fmt='d')
plt.show()