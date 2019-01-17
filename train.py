import keras
from keras import backend as K
from keras.models import save_model
import tensorflow as tf
from tf_utils_adv import tf_train, tf_test_error_rate
from fashion_mnist import *

flags = tf.app.flags
FLAGS = flags.FLAGS

K.set_image_data_format('channels_first')

def main(model_name, model_type):
    np.random.seed(0)
    assert keras.backend.backend() == "tensorflow"
    set_flags(64)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    
    flags.DEFINE_integer('NUM_EPOCHS', args.epochs, 'Number of epochs')
    flags.DEFINE_integer('type', args.type, 'model type')

    # Get fashion_mnist test data
    X_train, Y_train, X_test, Y_test = load_data()

    data_gen = data_flow(X_train)

    x = K.placeholder((None,
                       FLAGS.NUM_CHANNELS,
                       FLAGS.IMAGE_ROWS,
                       FLAGS.IMAGE_COLS
                       ))

    y = K.placeholder(shape=(None, FLAGS.NUM_CLASSES))

    model = model_select(type=model_type)

    # Train
    tf_train(x, y, model, X_train, Y_train, data_gen, model_name)

    # Finally print the result!
    test_error = tf_test_error_rate(model, x, X_test, Y_test)
    with open(model_name + '_log.txt', 'a') as log:
        log.write('Test error: %.1f%%' % test_error)
    print('Test error: %.1f%%' % test_error)
    save_model(model, model_name)
    json_string = model.to_json()
    with open(model_name+'.json', 'w') as f:
        f.write(json_string)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="path to model")
    parser.add_argument("--type", type=int, help="model type", default=0)
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs: fashion_mnist:50, svhn: 50 , cifar10: 150, cifar100: 200")
    args = parser.parse_args()

    main(args.model, args.type)
