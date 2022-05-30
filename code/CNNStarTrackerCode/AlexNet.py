import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

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
    target_size=(616,820),
    batch_size=32,
    class_mode='categorical'
)

#TEST
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    path+'/test',
    target_size=(616,820),
    batch_size=32,
    class_mode='categorical'
)


#BUILDING THE CONVOLUTIONAL NEURAL NETWORK
cnn = tf.keras.models.Sequential() #Sequence of layers
#CONVOLUTION 1
cnn.add(tf.keras.layers.Conv2D(filters=96,kernel_size=(11,11),strides=4,input_shape=[616,820,3]))
cnn.add(tf.keras.layers.LeakyReLU(alpha=0.05))
#POOLING 1
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2))
#CONVOLUTION 2
cnn.add(tf.keras.layers.Conv2D(filters=256,kernel_size=(5,5),strides=1))
cnn.add(tf.keras.layers.LeakyReLU(alpha=0.05))
#POOLING 2
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2))
#CONVOLUTION 3
cnn.add(tf.keras.layers.Conv2D(filters=384,kernel_size=(3,3),strides=1))
cnn.add(tf.keras.layers.LeakyReLU(alpha=0.05))
#CONVOLUTION 4
cnn.add(tf.keras.layers.Conv2D(filters=384,kernel_size=(3,3),strides=1))
cnn.add(tf.keras.layers.LeakyReLU(alpha=0.05))
#CONVOLUTION 5
cnn.add(tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),strides=1))
cnn.add(tf.keras.layers.LeakyReLU(alpha=0.05))
#POOLING 3
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2))

#FLATTENING
cnn.add(tf.keras.layers.Flatten())
#FULL CONNECTION
cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Dense(units=4096))
cnn.add(tf.keras.layers.LeakyReLU(alpha=0.05))
cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Dense(units=4096))
cnn.add(tf.keras.layers.LeakyReLU(alpha=0.05))
cnn.add(tf.keras.layers.Dense(units=1000))
cnn.add(tf.keras.layers.LeakyReLU(alpha=0.05))
cnn.add(tf.keras.layers.Dense(480,activation='softmax'))

#TRAINING THE CONVOLUTIONAL NEURAL NETWORK
#Compiling the CNN
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#Training the CNN on the Training set and evaluating it on the Test set
history = cnn.fit(x=training_set,validation_data=test_set,epochs=50)

#SAVING THE MODEL
cnn.save('./Results/Trained_Mini_StarTracker_AlexNet.h5')

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./Results/accuracyAlexNet.pdf')

plt.clf()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./Results/lossAlexNet.pdf')