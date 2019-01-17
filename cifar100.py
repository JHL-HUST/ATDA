from keras import backend as K
import keras
from keras.datasets import cifar100
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Input, GlobalAveragePooling2D, Lambda
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras_contrib.layers.normalization import GroupNormalization                                                    
import numpy as np
K.set_image_data_format('channels_first')


from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


def set_flags(batch_size):
    flags.DEFINE_integer('BATCH_SIZE', batch_size, 'Size of training batches')

    flags.DEFINE_integer('NUM_CLASSES', 100, 'Number of classification classes')
    flags.DEFINE_integer('IMAGE_ROWS', 32, 'Input row dimension')
    flags.DEFINE_integer('IMAGE_COLS', 32, 'Input column dimension')
    flags.DEFINE_integer('NUM_CHANNELS', 3, 'Input depth dimension')



def load_data(one_hot=True):
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    print("Loaded CIFAR-100 dataset.")

    if one_hot:
        # convert class vectors to binary class matrices
        y_train = np_utils.to_categorical(y_train, FLAGS.NUM_CLASSES).astype(np.float32)
        y_test = np_utils.to_categorical(y_test, FLAGS.NUM_CLASSES).astype(np.float32)

    return X_train, y_train, X_test, y_test



def modelZ():
    model = Sequential()
    model.add(Conv2D(96, (3, 3), padding = 'same', input_shape=(FLAGS.NUM_CHANNELS, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS)))
    model.add(GroupNormalization(axis=1))
    model.add(Activation('elu'))
    
    model.add(Conv2D(96, (3, 3), padding = 'same'))
    model.add(GroupNormalization(axis=1))
    model.add(Activation('elu'))
    model.add(Conv2D(96, (3, 3), padding = 'same', strides = 2))
    model.add(GroupNormalization(axis=1))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(192, (3, 3) , padding = 'same'))
    model.add(GroupNormalization(axis=1))
    model.add(Activation('elu'))
    model.add(Conv2D(192, (3, 3),  padding = 'same'))
    model.add(GroupNormalization(axis=1))
    model.add(Activation('elu'))
    model.add(Conv2D(192, (3, 3),  padding = 'same', strides = 2)) 
    model.add(GroupNormalization(axis=1))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(GroupNormalization(axis=1))
    model.add(Activation('elu'))
    model.add(Conv2D(192, (1, 1),padding='valid'))
    model.add(GroupNormalization(axis=1))
    model.add(Activation('elu'))
    model.add(Conv2D(100, (1, 1), padding='valid'))

    model.add(GlobalAveragePooling2D())
    return model



def modelA():
    weight_decay  = 0.0001
    model = Sequential()

    model.add(Conv2D(96, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal", input_shape=(FLAGS.NUM_CHANNELS, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS)))
    model.add(Activation('elu'))
    model.add(Conv2D(96, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same'))
  
    model.add(Dropout(0.5))
  
    model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
    model.add(Activation('elu'))
    model.add(Conv2D(192, (1, 1),padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same'))
  
    model.add(Dropout(0.5))
  
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
    model.add(Activation('elu'))
    model.add(Conv2D(256, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
    model.add(Activation('elu'))
    model.add(Conv2D(FLAGS.NUM_CLASSES, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
  
    model.add(GlobalAveragePooling2D())
    return model

def modelB():
    weight_decay  = 0.0001
    model = Sequential()

    model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal", input_shape=(FLAGS.NUM_CHANNELS, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS)))
    model.add(Activation('elu'))
    model.add(Conv2D(96, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same'))

    model.add(Dropout(0.5))

    model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
    model.add(Activation('elu'))
    model.add(Conv2D(192, (1, 1),padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same'))

    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
    model.add(Activation('elu'))
    model.add(Conv2D(256, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
    model.add(Activation('elu'))
    model.add(Conv2D(FLAGS.NUM_CLASSES, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
    model.add(GlobalAveragePooling2D())
    return model


def modelC():
    model = Sequential()
    model.add(Conv2D(96, (3, 3), activation='elu', padding = 'same', input_shape=(FLAGS.NUM_CHANNELS, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS)))
    model.add(Dropout(0.2))

    model.add(Conv2D(96, (3, 3), activation='elu', padding = 'same'))
    model.add(Conv2D(96, (3, 3), activation='elu', padding = 'same', strides = 2))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), activation='elu', padding = 'same'))
    model.add(Conv2D(192, (3, 3), activation='elu', padding = 'same', strides = 2))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), padding = 'same'))
    model.add(Activation('elu'))
    model.add(Conv2D(256, (1, 1),padding='valid'))
    model.add(Activation('elu'))
    model.add(Conv2D(FLAGS.NUM_CLASSES, (1, 1), padding='valid'))

    model.add(GlobalAveragePooling2D())

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
