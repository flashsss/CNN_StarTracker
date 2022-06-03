import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.applications.vgg16 import preprocess_input

train_path = './dataset/multitriangle/train'
test_path = './dataset/multitriangle/test'

resnet_model = Sequential()

pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(224,224,3),
                   pooling='avg',classes=480,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False

resnet_model.add(pretrained_model)
resnet_model.add(tf.keras.layers.Flatten())
resnet_model.add(tf.keras.layers.Dense(units=512, activation='relu'))
resnet_model.add(tf.keras.layers.Dense(units=480, activation='softmax'))
resnet_model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

from datetime import datetime

start = datetime.now()

history = resnet_model.fit(train_set, validation_data=test_set, epochs=50)

duration = datetime.now() - start
print("Training completed in time: ", duration)

resnet_model.save('./Results/ResNet_model.h5')

# summarize history for accuracy
plt.plot(resnet_model.history['accuracy'])
plt.plot(resnet_model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./Results/ResNetaccuracy.pdf')

plt.clf()

# summarize history for loss
plt.plot(resnet_model.history['loss'])
plt.plot(resnet_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./Results/ResNetloss.pdf')