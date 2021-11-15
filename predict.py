import sys
source = sys.argv[1]
if source == "help":
    print("Usage for this script is 'predict.py source_file.pgm model.h5'")
    exit()
import cv2
import numpy as np
from progressbar import ProgressBar
import tensorflow as tf
from tensorflow import keras
import warnings


c_model = str(sys.argv[2])
c_classes=c_model.layers[-1].output_shape[-1]
if c_classes == 3:
    continue
elif c_classes == 1:
    c_classes = 2
split_size_from_model=c_model.layers[0].output_shape[0][1]
warnings.filterwarnings("ignore")
def dice_coef_3cat(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 3 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = tf.keras.layers.Flatten()(tf.one_hot(tf.cast(y_true, 'int32'), 3)[...,1:])
    y_pred_f = tf.keras.layers.Flatten()(y_pred[...,1:])
    intersect = tf.math.reduce_sum(y_true_f * y_pred_f, -1)
    denom = tf.math.reduce_sum(y_true_f + y_pred_f, -1)
    return tf.math.reduce_mean((2. * intersect / (denom + smooth)))

def dice_coef_3cat_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef_3cat(y_true, y_pred)

def dice_coef_2cat(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for2 categories. Ignores background pixel label 0
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
    return 1 - dice_coef_3cat(y_true, y_pred)
def import_model(MODEL):

    model = tf.keras.models.load_model('./'+MODEL, custom_objects={'dice_coef_3cat_loss': dice_coef_3cat_loss})
    # model.summary()
    return model

"""
Cette fonction donne les points de coupe suivant la taille des imagettes
"""
def start_points(size, split_size, overlap=0.1):
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


def image_crop(INPUT, OVERLAP=0.5, WRITE_SINGLE_IMG=False):
    count = 0
    frmt = 'pgm'
    splitted = []
    name = 'split'
    img = cv2.imread(INPUT, -1)
    img = np.expand_dims(img, axis=-1)
    img_h, img_w, _ = img.shape
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    split_width = split_size_from_model
    split_height = split_size_from_model
    X_points = start_points(img_w, split_width, OVERLAP)
    Y_points = start_points(img_h, split_height, OVERLAP)
    for i in Y_points:
        for j in X_points:
            split = img[i:i + split_height, j:j + split_width]
            if WRITE_SINGLE_IMG:
                cv2.imwrite('splitted/{}_{}.{}'.format(name, str(count).zfill(5), frmt), split)
            splitted.append(split)
            count += 1
    splitted = np.array(splitted)
    return splitted


# RECONSTRUCTION


def avengers_assemble(CLASSES, INPUTS, OUTPUT="stitched", OVERLAP=0.5):
    if CLASSES != 3 and CLASSES != 2:
        print("Mauvais nombre de classes")
        return -1
    nbClasses = CLASSES
    if CLASSES == 2:
        nbClasses = 1
    results = INPUTS
    pbar = ProgressBar().start()
    # step1
    final_image = np.zeros((1104, 1104, nbClasses))
    final_image.fill(42)
    index = 0
    img_h, img_w = 1104, 1104
    split_width = split_size_from_model
    split_height = split_size_from_model
    X_points = start_points(img_w, split_width, OVERLAP)
    Y_points = start_points(img_h, split_height, OVERLAP)
    total_steps = 2 + len(Y_points) * len(X_points)
    pbar.update((1 / total_steps) * 100)
    # step2

    for i in Y_points:
        for j in X_points:
            # ici iront les 329889782932 conditions de superposition
            for k in range(split_size_from_model):
                for l in range(split_size_from_model):
                    """on cherche le max pour chaque image
                    si correspondance ok, sinon on prend celui de plus haute proba"""
                    if (final_image[i + k, j + l, 0] == 42):
                        for m in range(nbClasses):
                            final_image[i + k, j + l, m] = results[index, k, l, m]
                    else:
                        # max de en place
                        oldChannels = final_image[i + k, j + l, 0:nbClasses]
                        oldMax = max(oldChannels)
                        oldMaxIndex = np.where(oldChannels == oldMax)[0][0]
                        # max de new
                        newChannels = results[index, k, l, 0:nbClasses]
                        newMax = max(newChannels)
                        newMaxIndex = np.where(newChannels == newMax)[0][0]
                        # comparaison
                        if (newMaxIndex > oldMaxIndex):
                            final_image[i + k, j + l, oldMaxIndex] = newMax
            pbar.update(((1 + index) / total_steps) * 100)
            index += 1

    output = np.zeros((1104, 1104, 1))
    pbar.update((2 / total_steps) * 100)
    # step3
    if nbClasses == 3:
        for i in range(img_h):
            for j in range(img_w):
                channels = final_image[i, j, 0:nbClasses]
                maxChan = max(channels)
                indexMax = np.where(channels == maxChan)[0][0]
                if (indexMax == 0):
                    output[i, j, 0] = 0
                elif (indexMax == 1):
                    output[i, j, 0] = 127
                elif (indexMax == 2):
                    output[i, j, 0] = 255
    else:
        # ya peut etre un souci ici mais pas eu le temps de tester
        for i in range(img_h):
            for j in range(img_w):
                if (final_image[i, j, 0] > 0.5):
                    output[i, j, 0] = 255
                else:
                    output[i, j, 0] = 0
    cv2.imwrite(OUTPUT + '.pgm', output)
    pbar.update(100)
    return 1


def pred(INPUT, MODEL, WRITE_SINGLE_PRED=False):
    model = import_model(MODEL)
    if model == -1:
        return -1
    x_pred = INPUT
    pred = model.predict(x_pred)
    count = 0
    if WRITE_SINGLE_PRED:
        for img in pred:
            cv2.imwrite('predicted/' + str(count).zfill(5) + ".pgm", img)
            count += 1
    return pred


def cellseg(INPUT, MODEL, CLASSES):
    splitted = image_crop(INPUT)
    prediction = pred(splitted, MODEL)
    name = str(INPUT[:-4]) + "_segmented"
    if avengers_assemble(CLASSES, prediction, name) == 1:
        print("Done")
        return
    print("Error")
    return -1


cellseg(source, c_model, c_classes)
