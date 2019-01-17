import numpy as np
import keras.backend as K
import tensorflow as tf
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


def linf_loss(X1, X2):
    return np.max(np.abs(X1 - X2), axis=(1, 2, 3))


def gen_adv_loss(logits, y, loss='logloss', mean=False):
    """
    Generate the loss function.
    """

    if loss == 'training':
        # use the model's output instead of the true labels to avoid
        # label leaking at training time
        y = K.cast(K.equal(logits, K.max(logits, 1, keepdims=True)), "float32")
        y = y / K.sum(y, 1, keepdims=True)
        out = K.categorical_crossentropy(y, logits, from_logits=True)
    elif loss == 'min_training':
        y = K.cast(K.equal(logits, K.min(logits, 1, keepdims=True)), "float32")
        y = y / K.sum(y, 1, keepdims=True)
        out = K.categorical_crossentropy(y, logits, from_logits=True)
    elif loss == 'logloss':
        out = K.categorical_crossentropy(y, logits, from_logits=True)
    else:
        raise ValueError("Unknown loss: {}".format(loss))

    if mean:
        out = K.mean(out)
    else:
        out = K.sum(out)
    return out


def gen_grad(x, logits, y, loss='logloss'):
    """
    Generate the gradient of the loss function.
    """

    adv_loss = gen_adv_loss(logits, y, loss)

    # Define gradient of loss wrt input
    grad = K.gradients(adv_loss, [x])[0]
    return grad



def get_grad_L1(x, logits):
    x_shape = x.get_shape().as_list()
    dims = x_shape[1]*x_shape[2]*x_shape[3]
    
    adv_loss = gen_adv_loss(logits, None, loss='training')
    grad = K.gradients(adv_loss, [x])[0]
   
    flatten_grad = K.reshape(grad, shape=[-1, dims])
    L1_grad = K.sum(K.abs(flatten_grad), axis=-1)
    print(L1_grad)
    return L1_grad
