# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
import os
#print(os.listdir("../input/histopathologic-cancer-detection/sample_submission.csv"))
d = pd.read_csv("../input/histopathologic-cancer-detection/train_labels.csv")#.to_dict()
#print(len(d))
anses = {}
for i in range(len(d)):
    anses[str(d["id"][i])] = int(d["label"][i])
#print(os.listdir("../input/histopathologic-cancer-detection/"))
path = "../input/histopathologic-cancer-detection/train/"
Class = os.listdir(path)
print(anses[Class[100][:len(Class[100])-4]])
data=[]
labels=[]

height = 96
width = 96
channels = 3
classes = 2
n_inputs = height * width * channels


#print(os.listdir("../input/gtsrb-german-traffic-sign/train/"))
for a in range(len(Class)):
    if(a%1000==0):
        print(a/len(Class))
        
    if a/len(Class)>0.30:
        break
    try:
        image=cv2.imread(path+Class[a])
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((height, width))
        data.append(np.array(size_image))
        labels.append(anses[str(Class[a])[:(len(Class[a])-4)]])
    except AttributeError:
        print(" ")
            


#Randomize the order of the input images

(X_train,X_val)=Cells[(int)(0.2*len(labels)):],Cells[:(int)(0.2*len(labels))]
X_train = X_train.astype('float32')/255 
X_val = X_val.astype('float32')/255
(y_train,y_val)=labels[(int)(0.2*len(labels)):],labels[:(int)(0.2*len(labels))]

#Using one hote encoding for the train and validation labels
from keras.utils import to_categorical
y_train = to_categorical(y_train, 2)
y_val = to_categorical(y_val, 2)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

#model = Sequential()
#model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
#model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
#model.add(MaxPool2D(pool_size=(2, 2)))
#model.add(Dropout(rate=0.25))
#model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
#model.add(MaxPool2D(pool_size=(2, 2)))
#model.add(Dropout(rate=0.25))
#model.add(Flatten())
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(rate=0.5))
#model.add(Dense(43, activation='softmax'))


model = Sequential()
model.add(Conv2D(30, (20, 20), input_shape=X_train.shape[1:], activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



#Compilation of the model
#model.compile(loss='categorical_crossentropy', optimizer='adam',   metrics=['accuracy'])
epochs = 20
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_val, y_val))

#Display of the accuracy and the loss values
import matplotlib.pyplot as plt

plt.figure(0)
plt.plot(history.history['acc'], label='training accuracy')
plt.plot(history.history['val_acc'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
Cells=np.array(data)
labels=np.array(labels)

s=np.arange(Cells.shape[0])
np.random.seed(2)
np.random.shuffle(s)
#Cells=Cells[s]
labels=labels[s]
#Cells.shape[0]
y_test=pd.read_csv("../input/histopathologic-cancer-detection/test_labels.csv")
labels=y_test['Path'].as_matrix()
y_test=y_test['ClassId'].values

data=[]

for f in labels:
    image=cv2.imread('../input/gtsrb-german-traffic-sign/test/'+f.replace('Test/', ''))
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((height, width))
    data.append(np.array(size_image))

X_test=np.array(data)
X_test = X_test.astype('float32')/255 
pred = model.predict_classes(X_test)
