# Import the necessary libraries

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

"""# CNN Main"""

#To prevent overfitting -> training on the same images
#DATA PREPROCESSING
#TRAIN
featureMethod = 'multitriangle'
path = './dataset/'+featureMethod
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
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

#BUILDING THE CONVOLUTIONAL NEURAL NETWORK
cnn = tf.keras.models.Sequential() #Sequence of layers
#CONVOLUTION 1
cnn.add(tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',input_shape=[224,224,3]))
#POOLING 1
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
#CONVOLUTION 2
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
#POOLING 2
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
#CONVOLUTION 3
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
#POOLING 3
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

#FLATTENING
cnn.add(tf.keras.layers.Flatten())
#FULL CONNECTION
cnn.add(tf.keras.layers.Dense(units=256,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
cnn.add(tf.keras.layers.Dense(480,activation='softmax'))

from datetime import datetime

start = datetime.now()
#TRAINING THE CONVOLUTIONAL NEURAL NETWORK
#Compiling the CNN
cnn.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001),metrics=['accuracy'])
#Training the CNN on the Training set and evaluating it on the Test set
history = cnn.fit(x=training_set,validation_data=test_set,epochs=50)

duration = datetime.now() - start
print("Training completed in time: ", duration)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./Results/accuracy.pdf')

plt.clf()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./Results/loss.pdf')


#SAVING THE MODEL
cnn.save('./Results/SimpleCNN_model.h5')