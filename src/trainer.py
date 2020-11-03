""" Utility functions. """
## NOTE: You do not need to modify this block but you will need to use it.
import numpy as np
import os
import random
import csv
import pickle
import random
import tensorflow as tf
from src.models import MAML
from src.dataloaders import OmniglotDataGenerator, OmniglotDataGeneratorMutExclusive, PoseDataGenerator

# limit GPU memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print('ERROR: {}'.format(e))

# outer training loop 
def outer_train_step(inp, model, optim, meta_batch_size=25, num_inner_updates=1):
    with tf.GradientTape(persistent=False) as outer_tape:
        result = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

        outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

        total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

    gradients = outer_tape.gradient(total_losses_ts[-1], model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))

    total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
    total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
    total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]

    return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts

# outer evaluation loop 
def outer_eval_step(inp, model, meta_batch_size=25, num_inner_updates=1):
    result = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

    outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

    total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
    total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

    total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
    total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]

    return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts  

# meta training function
def meta_train_fn(model, exp_string, data_generator,
                   n_way=10, meta_train_iterations=15000, meta_batch_size=25,
                   k_shot=1, num_inner_updates=1, meta_lr=0.001, modeldir='models', 
                   log=True, logdir='logs', logfile='log.csv', dataset='omniglot'):
    SUMMARY_INTERVAL = 10
    SAVE_INTERVAL = 100
    PRINT_INTERVAL = SUMMARY_INTERVAL  
    TEST_PRINT_INTERVAL = SUMMARY_INTERVAL
    LOG_CSV_INTERVAL = SUMMARY_INTERVAL

    pre_losses, post_losses = [], []
    pre_accuracies, post_accuracies = [], []
    optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr)

    # make directories
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    
    # csv writer to log accuracies
    print('Logging to -> {}'.format(os.path.join(logdir,logfile)))
    with open(os.path.join(logdir,logfile), 'w', newline='') as csvfile:
        if dataset == 'omniglot':
            fieldnames = ['iter',                        \
                            'task_train_acc_pre_optim',  \
                            'task_train_acc_post_optim', \
                            'task_test_acc_pre_optim',   \
                            'task_test_acc_post_optim'   ]
        else:
            fieldnames = ['iter',                         \
                            'task_train_loss_pre_optim',  \
                            'task_train_loss_post_optim', \
                            'task_test_loss_pre_optim',   \
                            'task_test_loss_post_optim'   ]
        # write header
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()

        # training and test accuracy
        task_train_acc_pre_optim, task_train_acc_post_optim, task_test_acc_pre_optim, task_test_acc_post_optim = 0., 0., 0., 0.
      
        for itr in range(meta_train_iterations):
            # sample a batch of training data and partition into
            # group a (input_tr, label_tr) and group b (input_ts, label_ts)
            input_, label_ = data_generator.sample_batch(batch_type='meta_train', batch_size=meta_batch_size, shuffle=False)
            # support set
            input_tr = np.reshape(input_[:,:,:k_shot,:], newshape=(-1, n_way*k_shot, data_generator.dim_input))
            input_ts = np.reshape(input_[:,:,k_shot:,:], newshape=(-1, n_way*k_shot, data_generator.dim_input))
            # query set
            label_tr = np.reshape(label_[:,:,:k_shot,:], newshape=(-1, n_way*k_shot, data_generator.dim_output))
            label_ts = np.reshape(label_[:,:,k_shot:,:], newshape=(-1, n_way*k_shot, data_generator.dim_output))

            inp = (input_tr, input_ts, label_tr, label_ts)

            result = outer_train_step(inp, model, optimizer, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

            if itr % SUMMARY_INTERVAL == 0:
                if dataset == 'omniglot':
                    pre_accuracies.append(result[-2])
                    post_accuracies.append(result[-1][-1])
                elif dataset == 'pose':
                    pre_losses.append(result[2])
                    post_losses.append(result[-3][-1])


            if (itr!=0) and itr % PRINT_INTERVAL == 0:
                if dataset == 'omniglot':
                    task_train_acc_pre_optim = np.mean(pre_accuracies)
                    task_train_acc_post_optim = np.mean(post_accuracies)
                    print_str = 'Iteration %d: pre-inner-loop train accuracy: %.5f, post-inner-loop test accuracy: %.5f' % (itr, np.mean(pre_accuracies), np.mean(post_accuracies))
                    print(print_str)
                    pre_losses, post_losses, pre_accuracies, post_accuracies = [], [], [], []
                elif dataset == 'pose':
                    task_train_loss_pre_optim = np.mean(pre_losses)
                    task_train_loss_post_optim = np.mean(post_losses)
                    print_str = 'Iteration %d: pre-inner-loop train loss: %.5f, post-inner-loop test loss: %.5f' % (itr, np.mean(task_train_loss_pre_optim), np.mean(task_train_loss_post_optim))
                    print(print_str)
                    pre_losses, post_losses, pre_accuracies, post_accuracies = [], [], [], []

            if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
                # sample a batch of validation data and partition into
                # training (input_tr, label_tr) and testing (input_ts, label_ts)
                input_, label_ = data_generator.sample_batch(batch_type='meta_val', batch_size=meta_batch_size, shuffle=False)
                # support set
                input_tr = np.reshape(input_[:,:,:k_shot,:], newshape=(-1, n_way*k_shot, data_generator.dim_input))
                input_ts = np.reshape(input_[:,:,k_shot:,:], newshape=(-1, n_way*k_shot, data_generator.dim_input))
                # query set
                label_tr = np.reshape(label_[:,:,:k_shot,:], newshape=(-1, n_way*k_shot, data_generator.dim_output))
                label_ts = np.reshape(label_[:,:,k_shot:,:], newshape=(-1, n_way*k_shot, data_generator.dim_output))

                inp = (input_tr, input_ts, label_tr, label_ts)
                result = outer_eval_step(inp, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

                if dataset == 'omniglot':
                    task_test_acc_pre_optim = result[-2].numpy()
                    task_test_acc_post_optim = result[-1][-1].numpy()
                    print('Meta-validation pre-inner-loop train accuracy: %.5f, meta-validation post-inner-loop test accuracy: %.5f' % (result[-2], result[-1][-1]))
                elif dataset == 'pose':
                    task_test_loss_pre_optim = result[2].numpy()
                    task_test_loss_post_optim = result[-3][-1].numpy()
                    print('Meta-validation pre-inner-loop train loss: %.5f, meta-validation post-inner-loop test loss: %.5f' % (task_test_loss_pre_optim, task_test_loss_post_optim))

            if (itr!=0) and itr % LOG_CSV_INTERVAL == 0: 
                if dataset == 'omniglot':
                    csv_writer.writerow({   'iter'                        : itr,                       \
                                            'task_train_acc_pre_optim'    : task_train_acc_pre_optim,  \
                                            'task_train_acc_post_optim'   : task_train_acc_post_optim, \
                                            'task_test_acc_pre_optim'     : task_test_acc_pre_optim,   \
                                            'task_test_acc_post_optim'    : task_test_acc_post_optim, })
                else:
                    csv_writer.writerow({   'iter'                        : itr,                       \
                                            'task_train_loss_pre_optim'   : task_train_loss_pre_optim,  \
                                            'task_train_loss_post_optim'  : task_train_loss_post_optim, \
                                            'task_test_loss_pre_optim'    : task_test_loss_pre_optim,   \
                                            'task_test_loss_post_optim'   : task_test_loss_post_optim, })

            if (itr!=0) and itr % SAVE_INTERVAL == 0:
                model_file = modeldir + '/' + exp_string +  '/model' + str(itr)
                print("Saving model to ", model_file)
                model.save_weights(model_file)

# calculated for omniglot
NUM_META_TEST_POINTS = 600

# meta testing function
def meta_test_fn(model, data_generator, n_way=10, meta_batch_size=25, k_shot=1,
              num_inner_updates=1, dataset='omniglot'):
    np.random.seed(1)
    random.seed(1)

    meta_test_losses, meta_test_accuracies = [], []

    for _ in range(NUM_META_TEST_POINTS):
        # sample a batch of test data and partition into
        # group a (input_tr, label_tr) and group b (input_ts, label_ts)
        input_, label_ = data_generator.sample_batch(batch_type='meta_test', batch_size=meta_batch_size, shuffle=True)
        # support set
        input_tr = np.reshape(input_[:,:,:k_shot,:], newshape=(-1, n_way*k_shot, data_generator.dim_input))
        input_ts = np.reshape(input_[:,:,k_shot:,:], newshape=(-1, n_way*k_shot, data_generator.dim_input))
        # query set
        label_tr = np.reshape(label_[:,:,:k_shot,:], newshape=(-1, n_way*k_shot, data_generator.dim_output))
        label_ts = np.reshape(label_[:,:,k_shot:,:], newshape=(-1, n_way*k_shot, data_generator.dim_output))

        inp = (input_tr, input_ts, label_tr, label_ts)
        result = outer_eval_step(inp, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

        if dataset == 'omniglot':
            meta_test_accuracies.append(result[-1][-1])
        elif dataset == 'pose':
            meta_test_losses.append(result[-3][-1])

    meta_test_metrics = None
    if dataset == 'omniglot':
        meta_test_metrics = np.array(meta_test_accuracies)
    elif dataset == 'pose':
        meta_test_metrics = np.array(meta_test_losses)

    means = np.mean(meta_test_metrics)
    stds = np.std(meta_test_metrics)
    ci95 = 1.96*stds/np.sqrt(NUM_META_TEST_POINTS)

    print('Mean meta-test accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

# main function to run MAML
def run_maml(n_way=10, k_shot=1, meta_batch_size=25, meta_lr=0.001,
             inner_update_lr=0.4, num_filters=32, num_inner_updates=1,
             meta_train_batch_size=25, learn_inner_update_lr=False,
             resume=False, resume_itr=0, modeldir='./models',  
             log=True, logdir='./logs', dataset='omniglot', data_path='./omniglot_resized',
             meta_train=True, meta_train_iterations=15000, meta_train_k_shot=-1,
             meta_train_inner_update_lr=-1, mutual_exclusive=False):

    data_generator = None
    # call data_generator and get data with k_shot*2 samples per class
    if mutual_exclusive == True:
        data_generator = OmniglotDataGeneratorMutExclusive(n_way, k_shot*2, n_way, k_shot*2, config={'data_folder': data_path})
    elif dataset == 'omniglot':
        data_generator = OmniglotDataGenerator(n_way, k_shot*2, n_way, k_shot*2, config={'data_folder': data_path})
    elif dataset == 'pose':
        data_generator = PoseDataGenerator(n_way, k_shot*2, n_way, k_shot*2, config={'data_folder': data_path})

    # set up MAML model
    dim_output = data_generator.dim_output
    dim_input = data_generator.dim_input
    loss_func_ = None
    if dataset == 'omniglot':
        loss_func_ = 'cross_entropy'
    elif dataset == 'pose':
        loss_func_ = 'mse'
    model = MAML(dim_input,
              dim_output,
              num_inner_updates=num_inner_updates,
              inner_update_lr=inner_update_lr,
              k_shot=k_shot,
              num_filters=num_filters,
              learn_inner_update_lr=learn_inner_update_lr,
              dataset=dataset)

    if meta_train_k_shot == -1:
        meta_train_k_shot = k_shot
    if meta_train_inner_update_lr == -1:
        meta_train_inner_update_lr = inner_update_lr

    exp_string = 'cls_'+str(n_way)+'.mbs_'+str(meta_train_batch_size) + '.k_shot_' + str(meta_train_k_shot) + \
                '.inner_numstep_' + str(num_inner_updates) + '.meta_lr_' + str(meta_lr) + '.inner_updatelr_' + str(meta_train_inner_update_lr) + \
                '.learn_inner_update_lr_' + str(learn_inner_update_lr)  + '.dataset_' + str(dataset) + \
                '.mutual_exclusive_' + str(mutual_exclusive)

    logfile = exp_string+'.csv'

    if meta_train:
        meta_train_fn(model, exp_string, data_generator,
                      n_way, meta_train_iterations, meta_batch_size, k_shot,
                      num_inner_updates, meta_lr, modeldir, log, logdir, logfile=logfile, dataset=dataset)
    else:
        meta_batch_size = 1

        model_file = tf.train.latest_checkpoint(modeldir + '/' + exp_string)
        print("Restoring model weights from ", model_file)
        model.load_weights(model_file)

        meta_test_fn(model, data_generator, n_way, meta_batch_size, k_shot, num_inner_updates, dataset=dataset)