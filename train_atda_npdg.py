import keras
from keras import backend as K
from tensorflow.python.platform import flags
from keras.models import save_model

from cifar10 import *
from tf_utils import tf_train, tf_test_error_rate
from attack_utils import gen_grad
from fgs import symbolic_fgs, symbolic_alpha_fgs
import tensorflow as tf

K.set_image_data_format('channels_first')
FLAGS = flags.FLAGS


def main(model_name, adv_model_names, model_type):
    np.random.seed(0)
    assert keras.backend.backend() == "tensorflow"
    set_flags(32)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    flags.DEFINE_integer('NUM_EPOCHS', args.epochs, 'Number of epochs')
    flags.DEFINE_integer('type', args.type, 'model type')

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = load_data()

    data_gen = data_flow(X_train)

    x = K.placeholder(shape=(None,
                             FLAGS.NUM_CHANNELS,
                             FLAGS.IMAGE_ROWS,
                             FLAGS.IMAGE_COLS))

    y = K.placeholder(shape=(FLAGS.BATCH_SIZE, FLAGS.NUM_CLASSES))

    eps = args.eps

    # if src_models is not None, we train on adversarial examples that come
    # from multiple models
    adv_models = [None] * len(adv_model_names)
    for i in range(len(adv_model_names)):
        adv_models[i] = load_model(adv_model_names[i])

    model = model_select(type=model_type)

    x_advs = [None] * (len(adv_models) + 1)
    

    for i, m in enumerate(adv_models + [model]):
        x_noise = x + tf.random_uniform(shape=[FLAGS.BATCH_SIZE, FLAGS.NUM_CHANNELS, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS], minval= -args.eps, maxval=args.eps)
        x_noise = tf.clip_by_value(x_noise, 0., 1.)
        for _ in range(args.k):
            logits = m(x_noise)
            grad = gen_grad(x_noise, logits, y, loss='logloss')
            x_noise =  K.stop_gradient(x_noise + args.eps / 4.0 * K.sign(grad))
            x_noise = tf.clip_by_value(x_noise, x - args.eps, x + args.eps)
            x_noise = tf.clip_by_value(x_noise, 0., 1.)
        x_advs[i] = x_noise

    # Train an MNIST model
    tf_train(x, y, model, X_train, Y_train, data_gen, model_name, x_advs=x_advs)

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
    parser.add_argument('adv_models', nargs='*',
                        help='path to adv model(s)')
    parser.add_argument("--type", type=int, help="model type", default=0)
    parser.add_argument("--epochs", type=int, default=150,
                        help="number of epochs")
    parser.add_argument("--eps", type=float, default=4./255,
                        help="FGS attack scale")
    parser.add_argument("--k", type=float, default=10,
                        help="steps")
    args = parser.parse_args()
    main(args.model, args.adv_models, args.type)
