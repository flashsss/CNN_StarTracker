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

vggmodel.compile(loss = "categorical_crossentropy", optimizer = "rmsprop", metrics=["accuracy"])
history = vggmodel.fit(x = training_set,validation_data=test_set,epochs=50)