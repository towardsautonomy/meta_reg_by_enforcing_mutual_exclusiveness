from src.utils import parse_args
from src.trainer import run_maml

# main function
if __name__ == '__main__':
    
    # parse arguments                
    args = parse_args()
    
    # set up meta train k-shot
    if args.meta_train_k_shot == -1:
        args.meta_train_k_shot = args.k_shot
    if args.meta_train_batch_size == -1:
        args.meta_train_batch_size = args.meta_batch_size

    # run MAML
    run_maml(n_way=args.n_way, k_shot=args.k_shot, meta_batch_size=args.meta_batch_size, meta_lr=args.meta_lr,
                 inner_update_lr=args.inner_update_lr, num_filters=args.num_filters, num_inner_updates=args.num_inner_updates,
                 meta_train_batch_size=args.meta_train_batch_size, learn_inner_update_lr=args.learn_inner_update_lr, modeldir=args.modeldir,  
                 logdir=args.logdir, dataset=args.dataset, data_path=args.data_path, meta_train=False,
                 meta_train_iterations=args.meta_train_iterations, meta_train_k_shot=args.meta_train_k_shot,
                 mutual_exclusive=args.mutual_exclusive)

