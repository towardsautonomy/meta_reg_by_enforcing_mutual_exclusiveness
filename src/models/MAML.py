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

def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred - label))

"""Convolutional layers used by MAML model."""
## NOTE: You do not need to modify this block but you will need to use it.
seed = 123
def conv_block(inp, cweight, bweight, bn=None, activation=tf.nn.relu, stride=1):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    strides = [1,stride,stride,1]

    conv_output = tf.nn.conv2d(input=inp, filters=cweight, strides=strides, padding='SAME') + bweight
    if bn is not None:
        conv_output = bn(conv_output)
    conv_output = activation(conv_output)
    return conv_output

class ConvDeepLayers(tf.keras.layers.Layer):
    def __init__(self, channels, dim_hidden, dim_output, img_size):
        super(ConvDeepLayers, self).__init__()
        self.channels = channels
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.img_size = img_size

        weights = {}

        dtype = tf.float32
        weight_initializer =  tf.keras.initializers.GlorotUniform()
        k = 3

        # encoder
        weights['en_conv1'] = tf.Variable(weight_initializer(shape=[k, k, self.channels, self.dim_hidden]), name='en_conv1', dtype=dtype)
        weights['en_b1'] = tf.Variable(tf.zeros([self.dim_hidden]), name='en_b1')
        weights['en_conv2'] = tf.Variable(weight_initializer(shape=[k, k, self.dim_hidden, self.dim_hidden]), name='en_conv2', dtype=dtype)
        weights['en_b2'] = tf.Variable(tf.zeros([self.dim_hidden]), name='en_b3')
        weights['en_conv3'] = tf.Variable(weight_initializer(shape=[k, k, self.dim_hidden, self.dim_hidden]), name='en_conv3', dtype=dtype)
        weights['en_b3'] = tf.Variable(tf.zeros([self.dim_hidden]), name='en_b3')

        # tail layers
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
        en1 = conv_block(inp, weights['en_conv1'], weights['en_b1'], stride=1)
        en2 = conv_block(en1, weights['en_conv2'], weights['en_b2'], stride=1)
        en3 = conv_block(en2, weights['en_conv3'], weights['en_b3'], stride=1)

        hidden1 = conv_block(en3, weights['conv1'], weights['b1'], self.bn1, stride=1)
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], self.bn2, stride=2)
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], self.bn3, stride=2)
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], self.bn4, stride=1)
        hidden4 = tf.reduce_mean(input_tensor=hidden4, axis=[1, 2])
        return tf.matmul(hidden4, weights['w5']) + weights['b5']

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
        weights['conv2'] = tf.Variable(weight_initializer(shape=[k, k, self.dim_hidden, self.dim_hidden]), name='conv2', dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b2')
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn2')
        weights['conv3'] = tf.Variable(weight_initializer(shape=[k, k, self.dim_hidden, self.dim_hidden]), name='conv3', dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b3')
        self.bn3 = tf.keras.layers.BatchNormalization(name='bn3')
        weights['conv4'] = tf.Variable(weight_initializer([k, k, self.dim_hidden, self.dim_hidden]), name='conv4', dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b4')
        self.bn4 = tf.keras.layers.BatchNormalization(name='bn4')
        weights['w5'] = tf.Variable(weight_initializer(shape=[self.dim_hidden, self.dim_output]), name='w5', dtype=dtype)
        weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        self.conv_weights = weights

    def call(self, inp, weights):
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])
        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], self.bn1, stride=1)
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], self.bn2, stride=1)
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], self.bn3, stride=1)
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], self.bn4, stride=1)
        hidden4 = tf.reduce_mean(input_tensor=hidden4, axis=[1, 2])
        return tf.matmul(hidden4, weights['w5']) + weights['b5']

class MAMLVanillaConvLayers(tf.keras.layers.Layer):
    """Construct conv weights."""
    def __init__(self, channels, dim_hidden, dim_output, img_size):
        super(MAMLVanillaConvLayers, self).__init__()
        self.channels = channels
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.img_size = img_size

        weights = {}

        dtype = tf.float32
        weight_initializer =  tf.keras.initializers.GlorotUniform()
        k = 3
        weights['en_conv1'] = tf.Variable(weight_initializer(shape=[3, 3, 1, 32]), name='en_conv1', dtype=dtype)
        weights['en_bias1'] = tf.Variable(tf.zeros([32]), name='en_bias1')

        weights['en_conv2'] = tf.Variable(weight_initializer(shape=[3, 3, 32, 48]), name='en_conv2', dtype=dtype)
        weights['en_bias2'] = tf.Variable(tf.zeros([48]), name='en_bias2')

        weights['en_conv3'] = tf.Variable(weight_initializer(shape=[3, 3, 48, 64]), name='en_conv3', dtype=dtype)
        weights['en_bias3'] = tf.Variable(tf.zeros([64]), name='en_bias3')

        weights['en_full1'] = tf.Variable(weight_initializer(shape=[4096, 196]), name='en_full1', dtype=dtype)
        weights['en_bias_full1'] = tf.Variable(tf.zeros([196]), name='en_bias_full1')
        self.flatten_layer = tf.keras.layers.Flatten(name="en_flatten")

        weights['conv1'] = tf.Variable(weight_initializer(shape=[k, k, self.channels, self.dim_hidden]), name='conv1', dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b1')
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        weights['conv2'] = tf.Variable(weight_initializer(shape=[k, k, self.dim_hidden, self.dim_hidden]), name='conv2', dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b2')
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn2')
        weights['conv3'] = tf.Variable(weight_initializer(shape=[k, k, self.dim_hidden, self.dim_hidden]), name='conv3', dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b3')
        self.bn3 = tf.keras.layers.BatchNormalization(name='bn3')
        weights['conv4'] = tf.Variable(weight_initializer([k, k, self.dim_hidden, self.dim_hidden]), name='conv4', dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b4')
        self.bn4 = tf.keras.layers.BatchNormalization(name='bn4')
        weights['w5'] = tf.Variable(weight_initializer(shape=[self.dim_hidden, self.dim_output]), name='w5', dtype=dtype)
        weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')

        self.conv_weights = weights

    def call(self, inp, weights):
        """Forward conv."""
        # reuse is for the normalization parameters.
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])
        en1 = conv_block(inp, weights['en_conv1'], weights['en_bias1'], stride=2)
        en2 = conv_block(en1, weights['en_conv2'], weights['en_bias2'], stride=2)
        pool1 = tf.nn.max_pool(en2, 2, 2, 'VALID')

        en3 = conv_block(pool1, weights['en_conv3'], weights['en_bias3'], stride=2)
        out0 = self.flatten_layer(en3)
        out1 = tf.nn.relu(
            tf.matmul(out0, weights['en_full1']) + weights['en_bias_full1'])

        out1 = tf.reshape(out1, [-1, 14, 14, 1])

        hidden1 = conv_block(out1, weights['conv1'], weights['b1'], self.bn1, stride=1)
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], self.bn2, stride=1)
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], self.bn3, stride=1)
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], self.bn4, stride=1)

        # last hidden layer is 6x6x64-ish, reshape to a vector
        hidden4 = tf.reduce_mean(hidden4, [1, 2])
        # ipdb.set_trace()
        return tf.matmul(hidden4, weights['w5']) + weights['b5']

"""MAML model code"""
class MAML(tf.keras.Model):
    def __init__(self, dim_input=1, dim_output=1,
               num_inner_updates=1,
               inner_update_lr=0.4, num_filters=32, k_shot=5, learn_inner_update_lr=False,
               dataset='omniglot'):
        super(MAML, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.inner_update_lr = inner_update_lr
        self.dataset = dataset

        if dataset == 'omniglot':
            self.loss_func = partial(cross_entropy_loss, k_shot=k_shot)
        elif dataset == 'pose':
            self.loss_func = partial(mse)
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
        if dataset == 'omniglot':
            self.conv_layers = ConvDeepLayers(self.channels, self.dim_hidden, self.dim_output, self.img_size)
        elif dataset == 'pose':
            self.conv_layers = MAMLVanillaConvLayers(self.channels, self.dim_hidden, self.dim_output, self.img_size)

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

                        if self.dataset == 'omniglot':
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

            task_en_params_flattened = []
            task_tail_params_flattened = []
            for key in weights_optimized.keys():
                if 'en' not in key: # get layers beyond encoder
                    task_tail_params_flattened.extend([tf.reshape(weights_optimized[key], [-1])])
                else: # get encoder layers
                    task_en_params_flattened.extend([tf.reshape(weights_optimized[key], [-1])])
            task_params_flattened = [task_en_params_flattened, task_tail_params_flattened]

            # compute accuracies after inner update step
            for j in range(num_inner_updates):
                if self.dataset == 'omniglot':
                    task_accuracies_ts.append(accuracy(tf.argmax(input=label_ts, axis=1), tf.argmax(input=tf.nn.softmax(task_outputs_ts[j]), axis=1)))
                else:
                    task_accuracies_ts.append(0.0)

            task_output = [task_output_tr_pre, task_outputs_ts, task_loss_tr_pre, task_losses_ts, task_accuracy_tr_pre, task_accuracies_ts, task_params_flattened]
            return task_output

        input_tr, input_ts, label_tr, label_ts = inp
        # initializing the graph with dummy inputs.
        dummy_result = task_inner_loop((tf.identity(input_tr[0]), 
                                        tf.identity(input_ts[0]), 
                                        tf.identity(label_tr[0]), 
                                        tf.identity(label_ts[0])),
                              meta_batch_size,
                              num_inner_updates)
        
        # data type for the model
        model_en_params_dtype = []
        model_tail_params_dtype = []
        for key in self.conv_layers.conv_weights.keys():
            if 'en' not in key: # get layers beyond encoder
                model_tail_params_dtype.extend([tf.reshape(self.conv_layers.conv_weights[key], [-1])])
            else: # get encoder layers
                model_en_params_dtype.extend([tf.reshape(self.conv_layers.conv_weights[key], [-1])])
        model_en_params_dtype = [tf.float32]*len(model_en_params_dtype)
        model_tail_params_dtype = [tf.float32]*len(model_tail_params_dtype)
        model_params_dtype = [model_en_params_dtype, model_tail_params_dtype]


        # define output data type
        out_dtype = [tf.float32, [tf.float32]*num_inner_updates, tf.float32, [tf.float32]*num_inner_updates]
        out_dtype.extend([tf.float32, [tf.float32]*num_inner_updates, model_params_dtype])

        # define inner loop function
        task_inner_loop_partial = partial(task_inner_loop, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)
        
        result = tf.map_fn(task_inner_loop_partial,
                        elems=(input_tr, input_ts, label_tr, label_ts),
                        dtype=out_dtype,
                        parallel_iterations=meta_batch_size)

        return result