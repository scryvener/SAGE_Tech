# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 01:45:53 2022

@author: Kenneth
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay




num_classes = 21
input_shape = (100, 100, 1)

# import data

x_train_1=np.load(r'K:\Tessarect_ImageTest\class_train_data_x.npy')
x_train_2=np.load(r'K:\Tessarect_ImageTest\class_train_data_x_2.npy')

x_train=np.concatenate((x_train_1,x_train_2))

y_train_1=np.load(r'K:\Tessarect_ImageTest\class_train_data_y.npy')

y_train=np.concatenate((y_train_1,y_train_1))

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)

#reshape y and set to int

y_train=y_train.astype(int)-1
#y_train=np.reshape(y_train,(y_train.shape[0],1))

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train)

#%%


data_augmentation=keras.Sequential([
        
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
#        layers.RandomWidth(factor=(-0.2,0.2),interpolation='area'),
#        layers.RandomCrop(80,80),
#        layers.RandomTranslation(height_factor=(-.5,.5),width_factor=(-.5,.5)),
        
        ]
)

@tf.autograph.experimental.do_not_convert
def make_model(input_shape,num_classes):
    
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    
    x=layers.Conv2D(4, kernel_size=(25, 25))(x)
    x=layers.MaxPooling2D((5,5),strides=3)(x)
    x=layers.BatchNormalization()(x)
    x=layers.Activation("relu")(x)
    
    x=layers.Conv2D(16, kernel_size=(5, 5))(x)
    x=layers.MaxPooling2D((3,3))(x)
    x=layers.BatchNormalization()(x)
    x=layers.Activation("relu")(x)
    
    x=layers.Conv2D(64, kernel_size=(2, 2))(x)
    x=layers.BatchNormalization()(x)
    x=layers.Activation("relu")(x)
    
    x=layers.Flatten()(x)
    
    x=layers.Dense(48,activation="relu")(x)
    x=layers.BatchNormalization()(x)
    
    x=layers.Dropout(0.5)(x)
    outputs=layers.Dense(num_classes,activation="softmax")(x)
    
    return keras.Model(inputs,outputs)
    

model=make_model(input_shape=input_shape,num_classes=num_classes)

model.summary()

batch_size = 64
epochs = 50

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)


class_weights={0:.012,1:.0454,2:.0454,3:.0454,4:.0454,5:.0454,6:.0454,
               7:.0454,8:.0454,9:.0454,10:.0454,11:.0454,12:.0454,13:.0454,
               14:.0454,15:.0454,16:.0454,17:.0454,18:.0454,19:.0454,20:.0454#.1444
               }


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001,verbose=1,)


model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), metrics=["accuracy"])

#model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2,class_weight=class_weights,callbacks=[reduce_lr])


score = model.evaluate(x_train, y_train, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

predict_train=model.predict(x_train)

#%%
#at 150+ is starting to overfit, test acc drops to 40%, diff between train and acc now almost 10% rather then just ~5
model.save_weights(r'K:\PVP Companion ImageRec Model\AlternateTrain_4.18.2022_50epochs')


model.load_weights(r'K:\PVP Companion ImageRec Model\Best_1.29.2021_150epochs')
#%%% testing "pristine" data
import cv2
test_images=np.zeros((37,100,100))
for i in range(1,38):
    path=r'K:\Tessarect_ImageTest\PVP Scoreboards Test Data\test_icon_'+str(i)+'.png'
    
    img=cv2.imread(path)
    
    img_gray=get_grayscale(img)

    test_images[i-1,:,:]=img_gray.astype("float32")/255

x_test=np.reshape(test_images,(37,100,100,1))
y_test=np.array([7,1,10,3,6,13,12,4,8,7,14,3,8,1,14,20,2,13,9,14,20,2,13,9,7,7,12,12,18,18,16,20,18,8,14,16,21])-1
y_test=keras.utils.to_categorical(y_test,21)

predict=model.predict(x_test)
score = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

t=np.argmax(predict,1)

