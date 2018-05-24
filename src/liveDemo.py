from statistics import mode
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization


#parameters for loading data and images
FACEDETECTIONPATH = '../models/haarcascade_frontalface_default.xml'
MODELPATH = '../models/fer2013-35-0_69_180519.h5'
emotions = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}
frame_window = 10
emotion_offsets = (20, 40)

#Model Architecture
model = Sequential()
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

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation('softmax'))

#Import pretrained models
model.load_weights(MODELPATH)

face_detection = cv2.CascadeClassifier(FACEDETECTIONPATH)

# getting input model shapes for inference
emotion_target_size = model.input_shape[1:3]

emotion_window = []

# starting video streaming
cv2.namedWindow('Facial Emotion Recognition')
video_capture = cv2.VideoCapture(0)
while True:
    #convert image
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    #detect face
    faces = face_detection.detectMultiScale(gray_image, 1.3, 5)

    #data preparation
    for face_coordinates in faces:
        x, y, width, height = face_coordinates
        x_off, y_off = emotion_offsets
        x1, x2, y1, y2 = (x - x_off, x + width + x_off, y - y_off, y + height + y_off)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        gray_face = gray_face.astype('float32')
        gray_face = gray_face / 255.0
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        #Emotion prediction
        emotion_prediction = model.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotions[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        #select color
        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'disgust':
            color = emotion_probability * np.asarray((155,205,155))
        elif emotion_text == 'fear':
            color = emotion_probability * np.asarray((255, 255, 255))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        #add rectangle and emotion to the screen
        x, y, w, h = face_coordinates
        cv2.rectangle(rgb_image, (x, y), (x + w, y + h), color, 2)

        x, y = face_coordinates[:2]
        cv2.putText(rgb_image, emotion_mode, (x + 0, y + (-45)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Facial Emotion Recognition', bgr_image)

    #exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
