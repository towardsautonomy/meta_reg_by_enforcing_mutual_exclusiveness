import matplotlib.pyplot as plt
import os
import csv
import numpy as np
from src.utils import moving_average, parse_args

# main function
if __name__ == '__main__':

    # number of iterations to plot
    n_iterations = 5000
    dataset = 'omniglot'
    plotdir = 'plots'
    
    csv_files = ['logs/cls_20.mbs_10.k_shot_1.inner_numstep_1.meta_lr_0.0025.inner_updatelr_0.04.learn_inner_update_lr_False.dataset_omniglot.mutual_exclusive_False.metareg_False.lambda_0.1.tau_1.0.csv',
                 'logs/cls_20.mbs_10.k_shot_1.inner_numstep_1.meta_lr_0.0025.inner_updatelr_0.04.learn_inner_update_lr_False.dataset_omniglot.mutual_exclusive_False.metareg_True.lambda_0.1.tau_1.0.csv',
                 'logs/cls_20.mbs_10.k_shot_1.inner_numstep_1.meta_lr_0.0025.inner_updatelr_0.04.learn_inner_update_lr_False.dataset_omniglot.mutual_exclusive_False.metareg_True.lambda_0.1.tau_2.0.csv',
                 'logs/cls_20.mbs_10.k_shot_1.inner_numstep_1.meta_lr_0.0025.inner_updatelr_0.04.learn_inner_update_lr_False.dataset_omniglot.mutual_exclusive_False.metareg_True.lambda_0.1.tau_3.0.csv',
                 'logs/cls_20.mbs_10.k_shot_1.inner_numstep_1.meta_lr_0.0025.inner_updatelr_0.04.learn_inner_update_lr_False.dataset_omniglot.mutual_exclusive_False.metareg_True.lambda_0.2.tau_3.0.csv',
                 'logs/cls_20.mbs_10.k_shot_1.inner_numstep_1.meta_lr_0.0025.inner_updatelr_0.04.learn_inner_update_lr_False.dataset_omniglot.mutual_exclusive_False.metareg_True.lambda_0.3.tau_3.0.csv',
                 'logs/cls_20.mbs_10.k_shot_1.inner_numstep_1.meta_lr_0.0025.inner_updatelr_0.04.learn_inner_update_lr_False.dataset_omniglot.mutual_exclusive_False.metareg_True.lambda_0.4.tau_3.0.csv',
                 'logs/cls_20.mbs_10.k_shot_1.inner_numstep_1.meta_lr_0.0025.inner_updatelr_0.04.learn_inner_update_lr_False.dataset_omniglot.mutual_exclusive_False.metareg_True.lambda_0.4.tau_4.0.csv']

    legends = ['post-optim accuracy (no meta-reg)',
               'post-optim accuracy (lambda_0.1, tau_1.0)',
               'post-optim accuracy (lambda_0.1, tau_2.0)',
               'post-optim accuracy (lambda_0.1, tau_3.0)',
               'post-optim accuracy (lambda_0.2, tau_3.0)',
               'post-optim accuracy (lambda_0.3, tau_3.0)',
               'post-optim accuracy (lambda_0.4, tau_3.0)',
               'post-optim accuracy (lambda_0.4, tau_4.0)']

    # plot filename
    os.system('mkdir -p {}'.format(plotdir))
    plot_fname = os.path.join(plotdir, '{}.{}.png'.format(
        'dataset_{}.{}-way.{}-shot.innerlr_{}.outerlr_{}'.format(dataset,20,1,0.04,0.001),
        'compare_mutual_exclusive'))

    # plot graph
    plt.figure(figsize=(16,16))

    # plot
    metric = None
    for csv_file in csv_files:
        iteration = [] 
        task_train_metric_pre_optim, task_train_metric_post_optim, task_test_metric_pre_optim, task_test_metric_post_optim = [], [], [], []

        with open(csv_file) as file:
            reader = csv.DictReader( file )
            for line in reader:
                if int(line['iter']) > n_iterations:
                    break
                iteration.append(int(line['iter']))
                if dataset == 'omniglot':
                    task_train_metric_pre_optim.append(float(line['task_train_acc_pre_optim']))
                    task_test_metric_pre_optim.append(float(line['task_test_acc_pre_optim']))
                    task_train_metric_post_optim.append(float(line['task_train_acc_post_optim']))
                    task_test_metric_post_optim.append(float(line['task_test_acc_post_optim']))
                else:
                    task_train_metric_pre_optim.append(float(line['task_train_loss_pre_optim']))
                    task_test_metric_pre_optim.append(float(line['task_test_loss_pre_optim']))
                    task_train_metric_post_optim.append(float(line['task_train_loss_post_optim']))
                    task_test_metric_post_optim.append(float(line['task_test_loss_post_optim']))

            if dataset == 'omniglot':
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
            # plt.subplot(2,1,1)
            # plt.plot(iteration, task_train_metric_pre_optim, alpha=0.3)
            # plt.plot(iteration, task_train_metric_pre_optim_avg)
            # plt.plot(iteration, task_train_metric_post_optim, alpha=0.3)
            # plt.plot(iteration, task_train_metric_post_optim_avg)

            # plt.subplot(2,1,2)
            # plt.plot(iteration, task_test_metric_pre_optim, alpha=0.3)
            # plt.plot(iteration, task_test_metric_pre_optim_avg)
            # plt.plot(iteration, task_test_metric_post_optim, alpha=0.3)
            plt.plot(iteration, task_test_metric_post_optim_avg)

    # plot graphs    
    # plt.subplot(2,1,1)
    # if dataset == 'omniglot':
    #     plt.ylim(0, 1)
    # plt.title('Training {}'.format(metric))
    # plt.grid()
    # plt.xlabel('iteration')
    # plt.legend(legends)
# 
    # plt.subplot(2,1,2)
    if dataset == 'omniglot':
        plt.ylim(0, 1)
    plt.title('Validation {}'.format(metric), fontsize=16)
    plt.grid()
    plt.xlabel('iteration', fontsize=16)
    plt.legend(legends, fontsize=16)

    # save plot to file
    plt.savefig(plot_fname)
    plt.show()
    plt.close()