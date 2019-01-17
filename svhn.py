from keras import backend as K
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Lambda
import svhn_dataset
import numpy as np
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
from keras_contrib.layers.normalization import GroupNormalization

def set_flags(batch_size):
    flags.DEFINE_integer('BATCH_SIZE', batch_size, 'Size of training batches')

    flags.DEFINE_integer('NUM_CLASSES', 10, 'Number of classification classes')
    flags.DEFINE_integer('IMAGE_ROWS', 32, 'Input row dimension')
    flags.DEFINE_integer('IMAGE_COLS', 32, 'Input column dimension')
    flags.DEFINE_integer('NUM_CHANNELS', 3, 'Input depth dimension')


def load_data(one_hot=True):
    (X_train, y_train), (X_test, y_test) = svhn_dataset.load_dataset('./svhn')

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    if one_hot:
        # convert class vectors to binary class matrices
        y_train = np_utils.to_categorical(y_train, FLAGS.NUM_CLASSES).astype(np.float32)
        y_test = np_utils.to_categorical(y_test, FLAGS.NUM_CLASSES).astype(np.float32)
    
    return X_train, y_train, X_test, y_test



def modelZ():
    model = Sequential()
    model.add(ZeroPadding2D(padding=(1, 1),input_shape=(FLAGS.NUM_CHANNELS,
                                                        FLAGS.IMAGE_ROWS,
                                                        FLAGS.IMAGE_COLS))) 
    model.add(Convolution2D(16, 4, strides=2,
                            padding='valid'))
    model.add(Activation('relu'))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(32, 4, strides=2))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(FLAGS.NUM_CLASSES))

    return model



def modelA():
    model = Sequential()
    model.add(Convolution2D(32, 5, strides=2,
                            padding='valid',
                            input_shape=(FLAGS.NUM_CHANNELS,
                                         FLAGS.IMAGE_ROWS,
                                         FLAGS.IMAGE_COLS)))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 5, strides=2))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(FLAGS.NUM_CLASSES))
    return model


def modelB():
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(FLAGS.NUM_CHANNELS,
                                        FLAGS.IMAGE_ROWS,
                                        FLAGS.IMAGE_COLS)))
    model.add(Convolution2D(32, 3, strides=2))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, strides=2))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(FLAGS.NUM_CLASSES))
    return model


def modelC():
    model = Sequential()

    model.add(Convolution2D(64, 3, strides=1,input_shape=(FLAGS.NUM_CHANNELS,
                                                          FLAGS.IMAGE_ROWS,
                                                          FLAGS.IMAGE_COLS)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(300, kernel_initializer="he_normal", activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, kernel_initializer="he_normal", activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(FLAGS.NUM_CLASSES))
    return model



def model_select(type=0):

    models = [modelZ, modelA, modelB, modelC]

    return models[type]()


def data_flow(X_train):
    datagen = ImageDataGenerator()

    datagen.fit(X_train)
    return datagen


def load_model(model_path, type=0):

    try:
        with open(model_path+'.json', 'r') as f:
            json_string = f.read()
            model = model_from_json(json_string)
    except IOError:
        model = model_select(type=type)

    model.load_weights(model_path)
    return model
