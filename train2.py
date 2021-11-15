import tensorflow as tf
import os
import numpy as np
import cv2
from skimage.io import imread, imshow
from skimage.transform import resize
from tensorflow import keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime
print("plop")
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
def step_decay(losses):
    if len(history.losses)>1:
        lrate=0.01*1/(1+0.1*len(history.losses))
        momentum=0.8
        decay_rate=2e-6
        return lrate
    else:
        lrate=0.01
        return lrate

#Callbacks
history=LossHistory()
lrate=LearningRateScheduler(step_decay)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

#Optimizer


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
#model = build_unet((128, 128, 1), 3)

#Data import
print("Importing data...")
X_paths = os.listdir('/space/storage/homes/vrv/cellseg-cuda/processed_data/new_data/originals')
y_paths = os.listdir('/space/storage/homes/vrv/cellseg-cuda/processed_data/new_data/labels_3_classes')
X_paths.sort()
y_paths.sort()
nb_img=len(X_paths)
img_height=128
img_width=128
X = np.zeros((nb_img, img_height, img_width,1), dtype=np.uint8)
y = np.zeros((nb_img, img_height, img_width,1), dtype=np.uint8)
for i in range(len(X_paths)) : 
    img = imread('/space/storage/homes/vrv/cellseg-cuda/processed_data/new_data/originals/' + X_paths[i])
    img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
    img = np.expand_dims(img, axis=-1)
    X[i]=img
for j in range(len(y_paths)) : 
    img = imread('/space/storage/homes/vrv/cellseg-cuda/processed_data/new_data/labels_3_classes/' + y_paths[j])
    img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
    img = np.expand_dims(img, axis=-1)
    y[j]=img
n = len(X)
r = int(n*0.8)

X_train = X[:r]
X_test = X[r+1:]

y_train = y[:r]
y_test = y[r+1:]
print("Import done.")

#Set LR
lr = 0.01
model=keras.models.load_model('best_0.h5')
#Train
for i in range(10):
    if i != 0 :
        model=keras.models.load_model('best_' + str(i-1) + '.h5')
    modelName='best_' + str(i) + '.h5'
    mc = ModelCheckpoint(modelName, monitor='val_loss', mode='min', save_best_only=True)
    lr=lr/(10+i)
    opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    model.fit(X_train, y_train, validation_split=0.1, batch_size=16, epochs=100, callbacks=[tensorboard_callback, mc, es, history],verbose=2)