import keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from attack_utils import gen_adv_loss, get_grad_L1, gen_grad
from loss import get_margin_loss, get_coral_loss, get_mmd_loss
import time
import sys
import tensorflow.contrib.slim as slim
import pickle

from fgs import symbolic_fgs, symbolic_alpha_fgs

FLAGS = flags.FLAGS
EVAL_FREQUENCY = 100


def batch_eval(tf_inputs, tf_outputs, numpy_inputs):
    """
    A helper function that computes a tensor on numpy inputs by batches.
    From: https://github.com/openai/cleverhans/blob/master/cleverhans/utils_tf.py
    """

    n = len(numpy_inputs)
    assert n > 0
    assert n == len(tf_inputs)
    m = numpy_inputs[0].shape[0]
    for i in range(1, n):
        assert numpy_inputs[i].shape[0] == m

    out = []
    for _ in tf_outputs:
        out.append([])

    for start in range(0, m, FLAGS.BATCH_SIZE):
        batch = start // FLAGS.BATCH_SIZE

        # Compute batch start and end indices
        start = batch * FLAGS.BATCH_SIZE
        end = start + FLAGS.BATCH_SIZE
        numpy_input_batches = [numpy_input[start:end]
                               for numpy_input in numpy_inputs]
        cur_batch_size = numpy_input_batches[0].shape[0]
        assert cur_batch_size <= FLAGS.BATCH_SIZE
        for e in numpy_input_batches:
            assert e.shape[0] == cur_batch_size

        feed_dict = dict(zip(tf_inputs, numpy_input_batches))
        feed_dict[K.learning_phase()] = 0
        numpy_output_batches = K.get_session().run(tf_outputs,
                                                   feed_dict=feed_dict)
        for e in numpy_output_batches:
            assert e.shape[0] == cur_batch_size, e.shape
        for out_elem, numpy_output_batch in zip(out, numpy_output_batches):
            out_elem.append(numpy_output_batch)

    out = [np.concatenate(x, axis=0) for x in out]
    for e in out:
        assert e.shape[0] == m, e.shape
    return out



def tf_train(x, y, model, X_train, Y_train, generator, model_name, x_advs=None, epochs=0):
    old_vars = set(tf.global_variables())
    train_size = Y_train.shape[0]

    idx = tf.placeholder(dtype=tf.int32)
    maxval = tf.placeholder(dtype=tf.float32) 
    x_adv_chosen = tf.stack(x_advs)[idx]
         
    logits = model(x)
    preds = K.softmax(logits)
    l1 = gen_adv_loss(logits, y, mean=True) 
       
    logits_adv = model(x_adv_chosen)
    preds_adv = K.softmax(logits_adv)
    l2 = gen_adv_loss(logits_adv, y, mean=True)
        
    coral_loss = get_coral_loss(logits, logits_adv)
    mmd_loss = get_mmd_loss(logits, logits_adv)
    	
    alpha_item = 0.1
    margin_loss, _ , centers_update_up = get_margin_loss(tf.concat([tf.argmax(tf.cast(y, tf.int32), axis=-1), tf.argmax(tf.cast(y, tf.int32), axis=-1)], axis=0), tf.concat([logits, logits_adv], axis=0), FLAGS.NUM_CLASSES,  alpha=alpha_item)
    lambda_item = 1./3
	
    loss = (l1+l2) + (coral_loss + mmd_loss + margin_loss)*lambda_item
		
    # add regulazation loss for layer weights if it exists
    if len(model.losses) != 0:
        loss = loss + tf.add_n(model.losses)

    with tf.control_dependencies([centers_update_up]):
        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
 

    # Run all the initializers to prepare the trainable parameters.
    K.get_session().run(tf.initialize_variables(
        set(tf.global_variables()) - old_vars))
    start_time = time.time()
    print('Initialized!')
    

    # Loop through training steps.
    num_steps = int(FLAGS.NUM_EPOCHS * train_size + FLAGS.BATCH_SIZE - 1) // FLAGS.BATCH_SIZE

    step = 0
    for (batch_data, batch_labels) \
            in generator.flow(X_train, Y_train, batch_size=FLAGS.BATCH_SIZE):

        if len(batch_data) < FLAGS.BATCH_SIZE:
            k = FLAGS.BATCH_SIZE - len(batch_data)
            batch_data = np.concatenate([batch_data, X_train[0:k]])
            batch_labels = np.concatenate([batch_labels, Y_train[0:k]])

        
        feed_dict = {x: batch_data,
                     y: batch_labels,
                     K.learning_phase(): 1}

        # choose source of adversarial examples at random
        # (for ensemble adversarial training with domain adaptation)
        feed_dict[idx] = np.random.randint(len(x_advs))
            
        # Run the graph
        _, curr_loss, curr_l1, curr_l2, curr_mmd, curr_coral, curr_margin_loss, curr_preds, curr_preds_adv,_ = \
            K.get_session().run([optimizer, loss, l1, l2, mmd_loss, coral_loss,  margin_loss,  preds, preds_adv]
                                + [model.updates], feed_dict=feed_dict)

								
        if step % EVAL_FREQUENCY == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            with open(model_name + '_log.txt', 'a') as log:
                log.write('Step %d (epoch %.2f), %.2f s \n' % (step, float(step) * FLAGS.BATCH_SIZE / train_size, elapsed_time))
                log.write('Minibatch loss: %.3f (%.3f, %.3f, %.3f, %.3f,  %.3f) \n' % (curr_loss, curr_l1, curr_l2, curr_mmd, curr_coral,  curr_margin_loss))
                log.write('Minibatch error: %.1f%%, %.1f%% \n' % (error_rate(curr_preds, batch_labels), error_rate(curr_preds_adv, batch_labels)))

            sys.stdout.flush()

        step += 1
        if step == num_steps:
            break


def tf_test_error_rate(model, x, X_test, y_test):
    """
    Compute test error.
    """
    assert len(X_test) == len(y_test)

    # Predictions for the test set
    logits = model(x)
    eval_prediction = K.softmax(logits)
   
    feature_dict = {}
    predictions, logits_test = batch_eval([x], [eval_prediction, logits], [X_test])
    #feature_dict['logits_test'] = logits_test
    #feature_dict['y_test'] = y_test
    #pickle.dump(feature_dict, open("logits2/Znet_ens_ifgs.pkl","wb"))
    return error_rate(predictions, y_test)


def error_rate(predictions, labels):
    """
    Return the error rate in percent.
    """

    assert len(predictions) == len(labels)
    return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
