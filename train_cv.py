import time
import os
import torch
import numpy as np
import shutil

from options.train_options import TrainOptions

from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from data.eeg_dataset import EEGDatasetDataloader

from models import create_model
from models import networks

from utils.visualizer import Visualizer, Statistics
from utils.distance_metrics import calculate_distance
from utils import eeg_tools

from layers.normalizer import DataNormalizer

import sys
sys.path.append('/')


def autoupdate_params(opt, param_to_update, tuner_params):
    """
    :param opt:
    :param param_to_update:
    
    :return:
    """

    params = {}
    opt_dict = vars(opt)
    for p in param_to_update:
        params[p] = opt_dict[p]

    assert len(param_to_update) == len(params.keys())

    params.update(tuner_params)
    for k, v in params.items():
        print('Updating opt.{} from {}'.format(k, getattr(opt, k)))
        setattr(opt, k, v)
        print('to {}'.format(getattr(opt, k)))

    return opt


if __name__ == '__main__':
    parser = TrainOptions()  # get training options
    opt = parser.parse(save_opt=False)
    all_patients = ['lk', 'zxl', 'yjh', 'tll', 'lmk', 'lxh', 'wzw']
    experiment_name_prefix = opt.name
    ae_prefix = opt.ae_name
    score_names = ['temporal', 'mag 0-4', 'mag 5-7', 'mag 8-15', 'mag 16-28', 'mag 0-28',
                   'psd 0-4', 'psd 5-7', 'psd 8-15', 'psd 16-28', 'psd 0-28',
                   'phase_dist_mean', 'rmse_mean', 'cos_mean', 'phase_dtw_mean', 'phase_rmse_mean',
                   'fake_mag_dtw_mean', 'fake_mag_rmse_mean']
    params_to_autotune = ['g_lr', 'd_lr', 'lambda_L1', 'netD']
    # home_dir = '/public/home/xlwang/'  # '/home/'  depends on your data
    home_dir = '/home/'
    # flip = [5, 10, 20, 30, 40]
    flip = [1]
    # flip_G = [1, 5, 10]
    # flip_D = [1, 5]

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    rank = 0
    torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=0, world_size=1)

    torch.cuda.set_device(opt.local_rank)
    # rank = int(os.environ["RANK"])
    nproc_per_node = 2
    normalizer_dir = os.path.join(home_dir, 'wyq/Infos/norm_args/')
    single_mode = False  # if True, only use one patient to experiment
    # normalizer_dir = '/home/hmq/Infos/norm_args/'

    # for p_idx in range(1, len(all_patients)):
    for p_idx in range(1):
        if opt.leave_out == 'none':
            opt.leave_out = all_patients[p_idx]
        else:
            p_idx = all_patients.index(opt.leave_out)
        opt.name = experiment_name_prefix + '_' + opt.leave_out
        opt.ae_name = ae_prefix + '_' + opt.leave_out
        if single_mode:
            dataset = EEGDatasetDataloader(opt, [opt.leave_out])
        else:
            # create a dataset given opt.dataset_mode and other options
            dataset = EEGDatasetDataloader(opt, all_patients[:p_idx] + all_patients[p_idx + 1:])
        dataset_size = len(dataset)  # get the number of images in the dataset.
        opt.save_latest_freq = dataset_size

        parser.save_options(opt)
        dataset_name = os.path.basename(opt.dataroot)

        print('The number of training images = %d' % dataset_size)
        normalizer_name = dataset_name + '_without_' + opt.leave_out
        if opt.is_IF:
            normalizer_name += '_IF'
        if opt.pghi:
            normalizer_name += '_pghi'
        if single_mode:
            normalizer_name += '_single'
        normalizer_name += '.npy'
        compute_normalizer = False if os.path.exists(os.path.join(normalizer_dir, normalizer_name)) else True
        normalizer = DataNormalizer(dataset, os.path.join(normalizer_dir, normalizer_name), compute_normalizer,
                                    domain=opt.domain, use_phase=not opt.pghi)
        print("DataNormalizer prepared.")

        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)  # regular setup: load and print networks; create schedulers
        model.set_normalizer(normalizer)

        visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
        total_iters_single = 0  # the total number of training iterations
        total_iters_all = 0
        statistics = Statistics(opt)  
        
        # score_names = metric.get_score_names()
        # score_names = ('Train Precision', 'Train Recall', 'Test Precision', 'Test Recall')
        score_list = []
        lambda_L1 = opt.lambda_L1
        min_scores = dict.fromkeys(score_names[1:], float('inf'))
        min_scores['default'] = 0
        
        # update_lambda_L1_freq = (opt.n_epochs + opt.n_epochs_decay) // 6
        # model.start_training_G()
        networks.setup_seed(6)
        model.flip_training_G()

        # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        for epoch in range(opt.epoch_count,
                           opt.n_epochs + opt.n_epochs_decay + 1):
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration
            epoch_iter_single = 0  # the number of training iterations in current epoch, reset to 0 every epoch
            epoch_iter_all = 0
            visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
            
            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if epoch_iter_all > opt.max_dataset_size:
                    break
                if total_iters_all % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters_single += opt.batch_size
                epoch_iter_single += opt.batch_size
                total_iters_all += opt.batch_size * nproc_per_node
                epoch_iter_all += opt.batch_size * nproc_per_node
                model.set_input(data)  # unpack data from dataset and apply preprocessing
                model.update_batch_idx(i)
                model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

                # print training losses and save logging information to the disk
                if rank == 0 and epoch_iter_all % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter_all, losses, t_comp, t_data, rank)

                # cache our latest model every <save_latest_freq> iterations
                if total_iters_all % opt.save_latest_freq == 0:
                    if rank == 0:
                        print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters_all))
                        save_suffix = 'iter_%d' % total_iters_all if opt.save_by_iter else 'latest'
                        model.save_networks(save_suffix)
                    torch.distributed.barrier()

                if rank == 0 and total_iters_all % opt.loss_freq == 0:
                    losses = model.get_current_losses()
                    statistics.save_plotted_losses(epoch, int(epoch_iter_all / opt.print_freq),
                                                   float(epoch_iter_all) / dataset_size, losses)

                iter_data_time = time.time()

            if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                if rank == 0:
                    print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters_all))
                    model.save_networks('latest')
                    model.save_networks(epoch)
                torch.distributed.barrier()

            if (epoch >= opt.epoch_ahead) and (epoch % opt.val_epoch_freq == 0):
                if rank == 0:
                    cmd = "python test_cv.py --dataroot " + opt.dataroot + \
                          " --direction BtoA --dataset_mode eeg --norm instance" \
                          + " --input_nc " + str(opt.input_nc) + " --output_nc " + str(opt.output_nc) \
                          + " --preprocess none --no_flip --gpu_ids 0,1 --phase val" \
                          + " --model " + opt.model + " --ae_name " + ae_prefix + " --name " + experiment_name_prefix \
                          + ' --n_blocks ' + str(opt.n_blocks) \
                          + ' --ndf ' + str(opt.ndf) + ' --ngf ' + str(opt.ngf) \
                          + ' --leave_out ' + opt.leave_out
                    if opt.is_IF:
                        cmd += ' --is_IF'
                    if opt.pghi:
                        cmd += ' --pghi'
                    os.system(cmd)
                    results = calculate_distance(os.path.join(opt.dataroot, 'A', 'val'),
                                                 os.path.join(home_dir, 'wyq/projects/E2SGAN/results/',
                                                              # '/public/home/xlwang/hmq/Projects/experiments/results/',
                                                              opt.name, 'val_latest', 'npys'), normalizer, method='GAN',
                                                 aggregate=True, is_IF=opt.is_IF, aux_normalizer=normalizer)
                    score_list.append(
                        [results['temporal_mean']] + list(results['mag_mean']) + list(results['psd_mean']) +
                        [results['phase_dist_mean'][-1], results['rmse_mean'], results['cos_mean'],
                         results['phase_dtw_mean'], results['phase_rmse_mean'], results['fake_mag_dtw_mean'],
                         results['fake_mag_rmse_mean']])
                    # statistics.save_score_plots(opt.val_epoch_freq, {'names': score_names, 'scores': score_list})
                torch.distributed.barrier()

            if rank == 0:
                print('Rank %d \t End of epoch %d / %d \t Time Taken: %d sec' % (
                    rank, epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
            model.update_learning_rate()  # update learning rates at the end of every epoch.
