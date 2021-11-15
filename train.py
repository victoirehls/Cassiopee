import tensorflow as tf
import os
import numpy as np
import math
import cv2
from skimage.io import imread, imshow
from skimage.transform import resize
from tensorflow import keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime
#LR decay
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = [1,1]
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        sd.append(step_decay(len(self.losses)))
        print('lr:', step_decay(len(self.losses)))
sd=[]
from tensorflow.keras.callbacks import LearningRateScheduler
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


#Callbacks
history=LossHistory()
lrate=LearningRateScheduler(step_decay)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
modelName='sgd_crossentropy_256_2.h5'
mc = ModelCheckpoint(modelName, monitor='val_loss', mode='min', save_best_only=True)
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
#Optimizer
opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
#Model
def conv_block(inputs, filters, pool=True):
    x = Conv2D(filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if pool == True:
        p = MaxPool2D((2, 2))(x)
        return x, p
    else:
        return x

def dice_coef_2cat(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 10 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = tf.keras.layers.Flatten()(tf.one_hot(tf.cast(y_true, 'int32'), 2)[...,1:])
    y_pred_f = tf.keras.layers.Flatten()(y_pred[...,1:])
    intersect = tf.math.reduce_sum(y_true_f * y_pred_f, -1)
    denom = tf.math.reduce_sum(y_true_f + y_pred_f, -1)
    return tf.math.reduce_mean((2. * intersect / (denom + smooth)))

def dice_coef_2cat_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef_2cat(y_true, y_pred)
def build_unet(shape, num_classes):
    inputs = Input(shape)

    """ Encoder """
    x1, p1 = conv_block(inputs, 16, pool=True)
    x2, p2 = conv_block(p1, 32, pool=True)
    x3, p3 = conv_block(p2, 48, pool=True)
    x4, p4 = conv_block(p3, 64, pool=True)

    """ Bridge """
    b1 = conv_block(p4, 128, pool=False)
    """ Decoder """
    u1 = UpSampling2D((2, 2), interpolation="bilinear")(b1)
    c1 = Concatenate()([u1, x4])
    x5 = conv_block(c1, 64, pool=False)
    u2 = UpSampling2D((2, 2), interpolation="bilinear")(x5)
    c2 = Concatenate()([u2, x3])
    x6 = conv_block(c2, 48, pool=False)
    u3 = UpSampling2D((2, 2), interpolation="bilinear")(x6)
    c3 = Concatenate()([u3, x2])
    x7 = conv_block(c3, 32, pool=False)
    u4 = UpSampling2D((2, 2), interpolation="bilinear")(x7)
    c4 = Concatenate()([u4, x1])
    x8 = conv_block(c4, 16, pool=False)
    """ Output layer """
    output = Conv2D(num_classes, 1, padding="same", activation="softmax")(x8)

    return Model(inputs, output)
model = build_unet((256, 256, 1), 2)

#Data import
print("Importing data...")
X = np.load("X_256.npy")
y = np.load("y_256.npy")
for i in range(y.shape[0]):
    np.place(y[i],y[i]>1,1)
n = len(X)
r = int(n*0.8)

X_train = X[:r]
X_test = X[r+1:]

y_train = y[:r]
y_test = y[r+1:]
print("Import done.")

#Compile
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics = ["accuracy"])

#Train
fit_hist = model.fit(X_train, y_train, validation_split=0.1, batch_size=16, epochs=150, callbacks=[mc, es, history, lrate],verbose=1)
np.save('./sgd_crossentropy_256_2', fit_hist.history)