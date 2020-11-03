import matplotlib.pyplot as plt
import os
import csv
import numpy as np
from src.utils import moving_average, parse_args
from src.trainer import run_maml

# main function
if __name__ == '__main__':
    
    # parse arguments                
    args = parse_args()
    
    # set up args
    if args.meta_train_k_shot == -1:
        args.meta_train_k_shot = args.k_shot
    if args.meta_train_batch_size == -1:
        args.meta_train_batch_size = args.meta_batch_size

    # number of iterations to plot
    n_iterations = 15000
    
    # Plot the graphs
    exp_string = 'cls_'+str(args.n_way)+'.mbs_'+str(args.meta_train_batch_size) + '.k_shot_' + str(args.meta_train_k_shot) + \
                    '.inner_numstep_' + str(args.num_inner_updates) + '.meta_lr_' + str(args.meta_lr) + '.inner_updatelr_' + str(args.inner_update_lr) + \
                    '.learn_inner_update_lr_' + str(args.learn_inner_update_lr) + '.dataset_' + str(args.dataset) + \
                    '.mutual_exclusive_' + str(args.mutual_exclusive)
    csv_file = '{}/{}.csv'.format(args.logdir, exp_string)

    legends = ['pre-optimization raw plot',
               'pre-optimization moving avarage',
               'post-optimization raw plot',
               'post-optimization moving avarage']

    # plot filename
    os.system('mkdir -p {}'.format(args.plotdir))
    plot_fname = os.path.join(args.plotdir, '{}.png'.format(exp_string))

    # plot graph
    plt.figure(figsize=(16,16))

    iteration = [] 
    task_train_metric_pre_optim, task_train_metric_post_optim, task_test_metric_pre_optim, task_test_metric_post_optim = [], [], [], []

    # plot
    metric = None
    with open(csv_file) as file:
        reader = csv.DictReader( file )
        for line in reader:
            if int(line['iter']) > n_iterations:
                break
            iteration.append(int(line['iter']))
            if args.dataset == 'omniglot':
                task_train_metric_pre_optim.append(float(line['task_train_acc_pre_optim']))
                task_test_metric_pre_optim.append(float(line['task_test_acc_pre_optim']))
                task_train_metric_post_optim.append(float(line['task_train_acc_post_optim']))
                task_test_metric_post_optim.append(float(line['task_test_acc_post_optim']))
            else:
                task_train_metric_pre_optim.append(float(line['task_train_loss_pre_optim']))
                task_test_metric_pre_optim.append(float(line['task_test_loss_pre_optim']))
                task_train_metric_post_optim.append(float(line['task_train_loss_post_optim']))
                task_test_metric_post_optim.append(float(line['task_test_loss_post_optim']))

        if args.dataset == 'omniglot':
            metric = 'Accuracy'
            # compute moving average
            task_train_metric_pre_optim_avg = moving_average(task_train_metric_pre_optim, 20)
            task_test_metric_pre_optim_avg = moving_average(task_test_metric_pre_optim, 20)
            task_train_metric_post_optim_avg = moving_average(task_train_metric_post_optim, 20)
            task_test_metric_post_optim_avg = moving_average(task_test_metric_post_optim, 20)
        else:
            metric = 'Loss'
            # compute moving average
            task_train_metric_pre_optim_avg = moving_average(task_train_metric_pre_optim, 20)
            task_test_metric_pre_optim_avg = moving_average(task_test_metric_pre_optim, 20)
            task_train_metric_post_optim_avg = moving_average(task_train_metric_post_optim, 20)
            task_test_metric_post_optim_avg = moving_average(task_test_metric_post_optim, 20)

    # plot graphs    
    plt.subplot(2,1,1)
    if args.dataset == 'omniglot':
        plt.ylim(0, 1)
    plt.title('Training {}'.format(metric))
    plt.grid()
    plt.xlabel('iteration')
    plt.plot(iteration, task_train_metric_pre_optim, alpha=0.3)
    plt.plot(iteration, task_train_metric_pre_optim_avg)
    plt.plot(iteration, task_train_metric_post_optim, alpha=0.3)
    plt.plot(iteration, task_train_metric_post_optim_avg)
    plt.legend(legends)

    plt.subplot(2,1,2)
    if args.dataset == 'omniglot':
        plt.ylim(0, 1)
    plt.title('Validation {}'.format(metric))
    plt.grid()
    plt.xlabel('iteration')
    plt.plot(iteration, task_test_metric_pre_optim, alpha=0.3)
    plt.plot(iteration, task_test_metric_pre_optim_avg)
    plt.plot(iteration, task_test_metric_post_optim, alpha=0.3)
    plt.plot(iteration, task_test_metric_post_optim_avg)
    plt.legend(legends)

    # save plot to file
    plt.savefig(plot_fname)
    plt.show()
    plt.close()