#Import the necessary libraries

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras import optimizers
from keras.preprocessing import image

"""# CNN Main"""

#To prevent overfitting -> training on the same images
#DATA PREPROCESSING
#TRAIN
featureMethod = 'jaring'
path = './dataset/'+featureMethod
train_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
    path+'/train',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

#TEST
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    path+'/test',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

from keras.applications.vgg16 import VGG16
vggmodel = VGG16(weights='imagenet', include_top=True)
vggmodel.summary()
for layers in (vggmodel.layers)[:19]:
    print(layers)
    layers.trainable = False
X= vggmodel.layers[-2].output
predictions = Dense(480, activation="softmax")(X)
model_final = Model(input = vggmodel, output = predictions)
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=40, verbose=1, mode='auto')
hist = model_final.fit_generator(generator= training_set, steps_per_epoch= 2, epochs= 50, validation_data= test_set, validation_steps=1, callbacks=[checkpoint,early])
