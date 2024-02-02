# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:17:14 2021

@author: rstem
"""
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
path = (r'C:\Users\tempker\Documents\School')
os.chdir(path)

df = pd.read_csv('model_data.csv')
df_nonZero = df.loc[df['Count'] != 0]
df_nonZero['target'] = str(1)
df_Zero = df.loc[df['Count'] == 0]
df_Zero['target'] = str(0)


df_Zero = df_Zero.sample(n=138)
df = df_nonZero.append(df_Zero)
df =df.sample(frac=1)

datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)



train_generator=datagen.flow_from_dataframe(
    dataframe = df,
    x_col = 'path',
    y_col = 'target',
    subset="training",
    target_size= (512,512),
    batch_size=8,
    shuffle=True,
    class_mode="binary")

valid_generator=datagen.flow_from_dataframe(
    dataframe = df,
    x_col = 'path',
    y_col = 'target',
    subset="validation",
    target_size= (512,512),
    batch_size=8,
    shuffle=True,
    class_mode="binary")

test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
dataframe=df,
x_col = 'path',
y_col=None,
batch_size=1,
seed=8,
shuffle=False,
class_mode=None,
target_size= (512,512))


# example of tending the vgg16 model

# load model without classifier layers
model = VGG16(include_top=False, input_shape=(512, 512, 3))
# add new classifier layers
conv1 = Conv2D(512,(2,2), strides = (2,2))(model.layers[-1].output)
conv2 =Conv2D(512,(2,2), strides = (2,2))(conv1)
flat1 = Flatten()(conv2)
# class1 = Dense(1024, activation='relu')(flat1)
output = Dense(1, activation='sigmoid')(flat1)
# define new model
model = Model(inputs=model.inputs, outputs=output)
# summarize
model.summary()
# opt = Adam(learning_rate= 1e-10)
opt = SGD(learning_rate= 1e-4, clipvalue = 0.5)

model.compile(loss='binary_crossentropy', optimizer=opt,metrics=["accuracy"])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=25
)
model.save(r'D:\PhD\Independent Study')

# model = tf.keras.models.load_model()

model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID)


STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

