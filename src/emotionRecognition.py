import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint


#Parameters
MODELBASEPATH = './../models/'
DATASETPATH = './../dataset/fer2013.csv'
num_classes = 7 # 0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'
batch_size = 64
epochs = 150
width, height = 48, 48
validation_split = 0.2

#Data preparation
data = pd.read_csv(DATASETPATH)
pixels = data['pixels'].tolist()
faces = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(width, height)
    face = cv2.resize(face.astype('uint8'), (width, height))
    faces.append(face.astype('float32'))
faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)
emotions = pd.get_dummies(data['emotion']).as_matrix()
faces = faces / 255.0
num_samples = len(faces)
num_train_samples = int((1 - validation_split)*num_samples)
train_images = faces[:num_train_samples]
train_labels = emotions[:num_train_samples]
val_images = faces[num_train_samples:]
val_labels = emotions[num_train_samples:]
val_data = (val_images, val_labels)


#Model Architecture
model = Sequential()

#Convolutional + Relu + Dropout Layers
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#Fully Connected Layers
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

#Image preprocessing
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)
datagen.fit(train_images)

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

#When the loss-function doesn't improve 3 times change the learning rate
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

#log the lear process to tesorboard
tensorboard = TensorBoard(log_dir='./../logs', histogram_freq=1, write_graph=True, write_images=True)

#after 7 epochs without an improvement of the loss-function stopp, to avoid overfitting
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=1, mode='auto')

#Save model after each epoch when the loss-function improved
model_name =  MODELBASEPATH + 'fer2013-.{epoch:02d}-{val_acc:.2f}.h5'
checkpointer = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True)

#Train the model
history = model.fit_generator(datagen.flow(train_images, train_labels, batch_size),
                                      steps_per_epoch=len(train_images) / batch_size, epochs=epochs, verbose=1,
                                      validation_data=val_data,
                                      callbacks=[lr_reducer, tensorboard, early_stopper, checkpointer])
