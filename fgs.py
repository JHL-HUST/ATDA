import tensorflow as tf
import keras.backend as K
from attack_utils import gen_grad


def symbolic_fgs(x, grad, eps=0.3, clipping=True, reverse=False):
    """
    FGSM attack.
    """

    # signed gradient
    normed_grad = K.sign(grad)

    # Multiply by constant epsilon
    scaled_grad = eps * normed_grad

    # Add perturbation to original example to obtain adversarial example
    if not reverse:
        adv_x = K.stop_gradient(x + scaled_grad)
    else:
        adv_x = K.stop_gradient(x - scaled_grad)

    if clipping:
        adv_x = K.clip(adv_x, 0, 1)
    return adv_x


def symbolic_alpha_fgs(x, grad, eps, alpha, clipping=True):
    """
    R+FGSM attack.
    """

    # signed gradient
    normed_grad = K.sign(grad)

    # Multiply by constant epsilon
    scaled_grad = (eps-alpha) * normed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = K.stop_gradient(x + scaled_grad)

    if clipping:
        adv_x = K.clip(adv_x, 0, 1)
    return adv_x

def iter_fgs(model, x, y, steps, eps, alpha):
    """
    PGD / I-FGSM attack.
    """

    adv_x = x
   
    # iteratively apply the FGSM with small step size
    for i in range(steps):
        logits = model(adv_x)
        grad = gen_grad(adv_x, logits, y)

        adv_x = symbolic_fgs(adv_x, grad, alpha, True)
        adv_x = tf.clip_by_value(adv_x, x-eps, x+eps)
    return adv_x



def momentum_fgs(model, x, y, eps):

    # parameters
    nb_iter = 10
    decay_factor = 1.0
    eps_iter = eps / 5.0
	
    # Initialize loop variables
    momentum = tf.zeros_like(x)
    adv_x = x


    for i in range(nb_iter):
        logits = model(adv_x)
        grad = gen_grad(adv_x, logits, y)
        
        # Normalize current gradient and add it to the accumulated gradient
        red_ind = list(range(1, len(grad.get_shape())))
        avoid_zero_div = tf.cast(1e-12, grad.dtype)
        grad = grad / tf.maximum(avoid_zero_div, tf.reduce_mean(tf.abs(grad), red_ind, keepdims=True))
        momentum = decay_factor * momentum + grad

        normalized_grad = tf.sign(momentum)
        # Update and clip adversarial example in current iteration
        scaled_grad = eps_iter * normalized_grad
        adv_x = adv_x + scaled_grad
        adv_x = x + tf.clip_by_value(adv_x-x, -eps, eps)

        adv_x = tf.clip_by_value(adv_x, 0., 1.0)

        adv_x = K.stop_gradient(adv_x)

    return adv_x
