import numpy as np
import tensorflow as tf
import keras.backend as K
from cifar10 import load_data, set_flags, load_model
from fgs import symbolic_fgs, iter_fgs, momentum_fgs
from attack_utils import gen_grad
from tf_utils import tf_test_error_rate, batch_eval
from os.path import basename

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

K.set_image_data_format('channels_first')


def main(attack, src_model_name, target_model_names):
    np.random.seed(0)
    tf.set_random_seed(0)

    set_flags(20)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    x = K.placeholder((None,
                       FLAGS.NUM_CHANNELS,
                       FLAGS.IMAGE_ROWS,
                       FLAGS.IMAGE_COLS
                       ))

    y = K.placeholder((None, FLAGS.NUM_CLASSES))

    _, _, X_test, Y_test = load_data()

    # source model for crafting adversarial examples
    src_model = load_model(src_model_name)

    # model(s) to target
    target_models = [None] * len(target_model_names)
    for i in range(len(target_model_names)):
        target_models[i] = load_model(target_model_names[i])

    # simply compute test error
    if attack == "test":
        err = tf_test_error_rate(src_model, x, X_test, Y_test)
        print('{}: {:.1f}'.format(basename(src_model_name), 100-err))

        for (name, target_model) in zip(target_model_names, target_models):
            err = tf_test_error_rate(target_model, x, X_test, Y_test)
            print('{}: {:.1f}'.format(basename(name), 100-err))
        return

    eps = args.eps

    # take the random step in the RAND+FGSM
    if attack == "rfgs":
        X_test = np.clip(
            X_test + args.alpha * np.sign(np.random.randn(*X_test.shape)),
            0.0, 1.0)
        eps -= args.alpha

    logits = src_model(x)
    grad = gen_grad(x, logits, y)

    # FGSM and RAND+FGSM one-shot attack
    if attack in ["fgs", "rfgs"]:
        adv_x = symbolic_fgs(x, grad, eps=eps)

    # iterative FGSM
    if attack == "pgd":
        adv_x = iter_fgs(src_model, x, y, steps=args.steps, eps=args.eps, alpha=args.eps/10.0)
    
    if attack == 'mim':
        adv_x = momentum_fgs(src_model, x, y, eps=args.eps)

    print('start')
    # compute the adversarial examples and evaluate
    X_adv = batch_eval([x, y], [adv_x], [X_test, Y_test])[0]
    print('-----done----')
    # white-box attack
    err = tf_test_error_rate(src_model, x, X_adv, Y_test)
    print('{}->{}: {:.1f}'.format(basename(src_model_name), basename(src_model_name), 100-err))

    # black-box attack
    for (name, target_model) in zip(target_model_names, target_models):
        err = tf_test_error_rate(target_model, x, X_adv, Y_test)
        print('{}->{}: {:.1f}'.format(basename(src_model_name), basename(name), 100-err))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("attack", help="name of attack",
                        choices=["test", "fgs", "pgd", "rfgs", "mim"])
    parser.add_argument("src_model", help="source model for attack")
    parser.add_argument('target_models', nargs='*',
                        help='path to target model(s)')
    parser.add_argument("--eps", type=float, default=4./255,
                        help="attack scale")
    parser.add_argument("--alpha", type=float, default=2./255,
                        help="RAND+FGSM random perturbation scale")
    parser.add_argument("--steps", type=int, default=20,
                        help="Iterated FGS steps for PGD")

    args = parser.parse_args()
    main(args.attack, args.src_model, args.target_models)
