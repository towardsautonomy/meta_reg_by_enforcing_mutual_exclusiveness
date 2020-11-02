import sys
import numpy as np
import tensorflow as tf
from functools import partial

## Loss utilities
def cross_entropy_loss(pred, label, k_shot):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=tf.stop_gradient(label)) / k_shot)

def accuracy(labels, predictions):
    return tf.reduce_mean(tf.cast(tf.equal(labels, predictions), dtype=tf.float32))

def l2_loss(pred, label):
    return tf.cast((tf.nn.l2_loss(pred - label) / label.shape[0] * label.shape[1]), dtype=tf.float32)

"""Convolutional layers used by MAML model."""
## NOTE: You do not need to modify this block but you will need to use it.
seed = 123
def conv_block(inp, cweight, bweight, bn, activation=tf.nn.relu, stride=1):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    strides = [1,stride,stride,1]

    conv_output = tf.nn.conv2d(input=inp, filters=cweight, strides=strides, padding='SAME') + bweight
    normed = bn(conv_output)
    normed = activation(normed)
    return normed

class ConvLayers(tf.keras.layers.Layer):
    def __init__(self, channels, dim_hidden, dim_output, img_size):
        super(ConvLayers, self).__init__()
        self.channels = channels
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.img_size = img_size

        weights = {}

        dtype = tf.float32
        weight_initializer =  tf.keras.initializers.GlorotUniform()
        k = 3

        weights['conv1'] = tf.Variable(weight_initializer(shape=[k, k, self.channels, self.dim_hidden]), name='conv1', dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b1')
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        weights['conv2'] = tf.Variable(weight_initializer(shape=[k, k, self.dim_hidden, 2*self.dim_hidden]), name='conv2', dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([2*self.dim_hidden]), name='b2')
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn2')
        weights['conv3'] = tf.Variable(weight_initializer(shape=[k, k, 2*self.dim_hidden, 4*self.dim_hidden]), name='conv3', dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([4*self.dim_hidden]), name='b3')
        self.bn3 = tf.keras.layers.BatchNormalization(name='bn3')
        weights['conv4'] = tf.Variable(weight_initializer([k, k, 4*self.dim_hidden, 4*self.dim_hidden]), name='conv4', dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([4*self.dim_hidden]), name='b4')
        self.bn4 = tf.keras.layers.BatchNormalization(name='bn4')
        weights['w5'] = tf.Variable(weight_initializer(shape=[4*self.dim_hidden, self.dim_output]), name='w5', dtype=dtype)
        weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        self.conv_weights = weights

    def call(self, inp, weights):
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])
        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], self.bn1, stride=1)
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], self.bn2, stride=2)
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], self.bn3, stride=2)
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], self.bn4, stride=1)
        hidden4 = tf.reduce_mean(input_tensor=hidden4, axis=[1, 2])
        return tf.matmul(hidden4, weights['w5']) + weights['b5']
    
"""MAML model code"""
class MAML(tf.keras.Model):
    def __init__(self, dim_input=1, dim_output=1,
               num_inner_updates=1,
               inner_update_lr=0.4, num_filters=32, k_shot=5, learn_inner_update_lr=False,
               loss_func='cross_entropy'):
        super(MAML, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.inner_update_lr = inner_update_lr
        self.loss_func_ = loss_func

        if self.loss_func_ == 'cross_entropy':
            self.loss_func = partial(cross_entropy_loss, k_shot=k_shot)
        elif self.loss_func_ == 'mse':
            self.loss_func = partial(l2_loss)
        self.dim_hidden = num_filters
        self.channels = 1
        self.img_size = int(np.sqrt(self.dim_input/self.channels))

        # outputs_ts[i] and losses_ts_post[i] are the output and loss after i+1 inner gradient updates
        losses_tr_pre, outputs_tr, losses_ts_post, outputs_ts = [], [], [], []

        # for each loop in the inner training loop
        outputs_ts = [[]]*num_inner_updates
        losses_ts_post = [[]]*num_inner_updates
        accuracies_ts = [[]]*num_inner_updates

        # Define the weights - these should NOT be directly modified by the
        # inner training loop
        tf.random.set_seed(seed)
        self.conv_layers = ConvLayers(self.channels, self.dim_hidden, self.dim_output, self.img_size)

        self.learn_inner_update_lr = learn_inner_update_lr
        if self.learn_inner_update_lr:
            self.inner_update_lr_dict = {}
            for key in self.conv_layers.conv_weights.keys():
                self.inner_update_lr_dict[key] = [tf.Variable(self.inner_update_lr, name='inner_update_lr_%s_%d' % (key, j)) for j in range(num_inner_updates)]
  
    @tf.function
    def call(self, inp, meta_batch_size=25, num_inner_updates=1):
        def task_inner_loop(inp, meta_batch_size=25, num_inner_updates=1):
            """
            Perform gradient descent for one task in the meta-batch (i.e. inner-loop).
            Args:
              inp: a tuple (input_tr, input_ts, label_tr, label_ts), where input_tr and label_tr are the inputs and
                labels used for calculating inner loop gradients and input_ts and label_ts are the inputs and
                labels used for evaluating the model after inner updates.
                Should be shapes:
                  input_tr: [N*K, flattened_img]
                  input_ts: [N*K, flattened_img]
                  label_tr: [N*K, output_dim]
                  label_ts: [N*K, output_dim]
            Returns:
              task_output: a list of outputs, losses and accuracies at each inner update
            """
            # the inner and outer loop data
            input_tr, input_ts, label_tr, label_ts = inp

            # weights corresponds to the initial weights in MAML (i.e. the meta-parameters)
            weights = self.conv_layers.conv_weights

            # the predicted outputs, loss values, and accuracy for the pre-update model (with the initial weights)
            # evaluated on the inner loop training data
            task_output_tr_pre, task_loss_tr_pre, task_accuracy_tr_pre = None, None, None

            # lists to keep track of outputs, losses, and accuracies of test data for each inner_update
            # where task_outputs_ts[i], task_losses_ts[i], task_accuracies_ts[i] are the output, loss, and accuracy
            # after i+1 inner gradient updates
            task_outputs_ts, task_losses_ts, task_accuracies_ts = [], [], []

            # list to track learned learning rate
            inner_update_lr_dict_ = []

            # perform num_inner_updates to get modified weights
            # modified weights should be used to evaluate performance
            # At each inner update, use input_tr and label_tr for calculating gradients
            # and use input_ts and labels for evaluating performance

            # make a copy of weights to optimize during inner update
            weights_optimized = dict(zip(weights.keys(), [weights[key] for key in weights.keys()]))

            with tf.GradientTape(persistent=True) as inner_tape:
                # perform inner loop updates
                for j in range(num_inner_updates):
                    # get logits for training data
                    logits_tr = self.conv_layers(input_tr, weights_optimized)
                    # compute training loss 
                    loss_tr_j = self.loss_func(logits_tr, label_tr)

                    # pre-optimization output and accuracy
                    if (task_output_tr_pre is None) and (task_loss_tr_pre is None) and (task_accuracy_tr_pre is None):
                        # compute pre-update outputs and losses
                        task_output_tr_pre = logits_tr
                        # Compute losses from output predictions
                        task_loss_tr_pre = loss_tr_j

                        if self.loss_func_ == 'cross_entropy':
                            # Compute accuracies from output predictions
                            task_accuracy_tr_pre = accuracy(tf.argmax(input=label_tr, axis=1), tf.argmax(input=tf.nn.softmax(task_output_tr_pre), axis=1))
                        else:
                            task_accuracy_tr_pre = 0.0

                    # compute gradients
                    grads = inner_tape.gradient(loss_tr_j, list(weights_optimized.values()))
                    # make a dictionary for gradients
                    grad_dict = dict(zip(weights_optimized.keys(), grads))

                    if self.learn_inner_update_lr:
                        # update optimized weights using learnable learning rate
                        weights_optimized = dict(zip(weights_optimized.keys(),
                                                  [weights_optimized[key] - self.inner_update_lr_dict[key][j] * grad_dict[key] for key in weights_optimized.keys()]))

                    else:
                        # update optimized weights
                        weights_optimized = dict(zip(weights_optimized.keys(),
                                              [weights_optimized[key] - self.inner_update_lr * grad_dict[key] for key in weights_optimized.keys()]))

                    # compute logits for test data
                    logits_ts = self.conv_layers(input_ts, weights_optimized)
                    # compute loss 
                    loss_ts_j = self.loss_func(logits_ts, label_ts)
                    # add to the task output list
                    task_outputs_ts.append(logits_ts)
                    # compute task loss on test data
                    task_losses_ts.append(loss_ts_j)

            # compute accuracies after inner update step
            for j in range(num_inner_updates):
                if self.loss_func_ == 'cross_entropy':
                    task_accuracies_ts.append(accuracy(tf.argmax(input=label_ts, axis=1), tf.argmax(input=tf.nn.softmax(task_outputs_ts[j]), axis=1)))
                else:
                    task_accuracies_ts.append(0.0)

            task_output = [task_output_tr_pre, task_outputs_ts, task_loss_tr_pre, task_losses_ts, task_accuracy_tr_pre, task_accuracies_ts]
            return task_output

        input_tr, input_ts, label_tr, label_ts = inp
        # initializing the graph with dummy inputs.
        dummy_result = task_inner_loop((tf.identity(input_tr[0]), 
                                        tf.identity(input_ts[0]), 
                                        tf.identity(label_tr[0]), 
                                        tf.identity(label_ts[0])),
                              meta_batch_size,
                              num_inner_updates)
        
        out_dtype = [tf.float32, [tf.float32]*num_inner_updates, tf.float32, [tf.float32]*num_inner_updates]
        out_dtype.extend([tf.float32, [tf.float32]*num_inner_updates])
        task_inner_loop_partial = partial(task_inner_loop, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)
        result = tf.map_fn(task_inner_loop_partial,
                        elems=(input_tr, input_ts, label_tr, label_ts),
                        dtype=out_dtype,
                        parallel_iterations=meta_batch_size)
        return result