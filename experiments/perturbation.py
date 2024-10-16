import numpy as np
import torch
from networkx.drawing.tests.test_pylab import mpl
from scipy.stats import pearsonr
from utils.distance_metrics import bestK_results
from layers.normalizer import DataNormalizer
import os
from options.test_options import TestOptions
from models import create_model
from utils import html
from utils.numpy_tools import save_origin_npy
import matplotlib.pyplot as plt
from data.eeg_dataset import EEGDatasetDataloader
import shutil
from utils.eeg_tools import IF_to_eeg
import mne
from dtaidistance import dtw

EEGTrainingDir = '/public/home/xlwang/hmq/Datasets/cv_0704_60/B/train'
SEEGTrainingDir = '/public/home/xlwang/hmq/Datasets/cv_0704_60/A/train'
resultDir = '/public/home/xlwang/hmq/Projects/experiments/results/cv_pix2pix_global_onesided_0908_tll/trainAsTest_36/npys'
normalizerDir = '/public/home/xlwang/hmq/Infos/norm_args/cv_0704_60_without_tll_IF.npy'
EEGPosDir = '/home/hmq/Infos/position_info/eeg_pos.npy'  # '/public/home/xlwang/hmq/Infos/position_info/eeg_pos.npy'
patientList = ['lk', 'zxl', 'yjh', 'lxh', 'wzw', 'lmk']  # no tll
freq_bands = {'delta': (0, 4), 'theta': (4, 8), 'alpha': (8, 16), 'beta': (16, 32)}


plt.rcParams.update({'font.size': 7})


def frequencywise_perturbation(original, low, high):

    freq_step = 128 // 32
    width = original.shape[0]
    std = original.std()
    gaussianNoise = np.random.normal(loc=0, scale=std, size=((high - low) * freq_step, width))
    original[low * freq_step: high * freq_step, :] += gaussianNoise

    return original, gaussianNoise.mean()


def perturbation_correlation(n_perturbation, normalizer, investigated, saveDir):

    correlations = dict.fromkeys(list(freq_bands.keys()), 0)
    pvalue = dict.fromkeys(list(freq_bands.keys()), 0)
    for k in correlations.keys():
        correlations[k] = {}
        pvalue[k] = {}

    parser = TestOptions()  # get training options
    opt = parser.parse(save_opt=False)
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    # opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    experiment_name_prefix = opt.name
    ae_prefix = opt.ae_name
    opt.name = experiment_name_prefix + '_' + opt.leave_out
    opt.ae_name = ae_prefix + '_' + opt.leave_out
    parser.save_options(opt)
    dataset_name = os.path.basename(opt.dataroot)

    # dataset = EEGDatasetDataloader(opt, patientList)  # create a dataset given opt.dataset_mode and other options
    dataset = EEGDatasetDataloader(opt, [investigated])
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    model.set_normalizer(normalizer)
    web_dir = os.path.join(opt.results_dir, opt.name,
                           '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    npy_dir = os.path.join(web_dir, "npys")
    if os.path.exists(npy_dir):
        shutil.rmtree(npy_dir)
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    print('Dataset size:{}'.format(len(dataset)))

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        # if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #     break
        f_n = os.path.basename(data['A_paths'][0])
        f_n = f_n.split('.')[0]
        f_n = f_n + '_fake_B.npy'
        patient = f_n.split('_')[0]
        EEG_chan = f_n.split('_')[1]
        if patient != investigated:
            continue
        for band_name in freq_bands.keys():
            if EEG_chan not in correlations[band_name].keys():
                correlations[band_name][EEG_chan] = []
                pvalue[band_name][EEG_chan] = []

        real = data['A'][0].numpy()
        real = IF_to_eeg(real, normalizer, iseeg=False, is_IF=True)[0].astype(np.float64)
        real_psd = mne.time_frequency.psd_array_welch(real, 64, n_fft=256)
        original_fake = np.load(os.path.join(resultDir, f_n))
        original_fake = IF_to_eeg(original_fake, normalizer, iseeg=False, is_IF=True)[0].astype(np.float64)
        original_psd = mne.time_frequency.psd_array_welch(original_fake, 64, n_fft=256)
        original_psd_dist = np.linalg.norm(np.asarray(real_psd[0]) - np.asarray(original_psd[0]), ord=2) / len(
            real_psd[0]) ** 0.5
        # original_dtw = dtw.distance_fast(real, original_fake, use_pruning=True)

        for band_name, band in freq_bands.items():
            mag_increment_ls = []
            psd_increment_ls = []
            # dtw_increment_ls = []
            for i in range(n_perturbation):
                perturbated, mag_increment = frequencywise_perturbation(data['B'][0][0].numpy(), band[0],
                                                                        band[1])  # perturbate input EEG
                data['B'][0][0] = torch.from_numpy(perturbated)
                model.set_input(data)  # unpack data from data loader
                model.test()  # run inference
                visuals = model.get_current_visuals()  # get image result
                perturbated_fake = visuals['fake_B'][0].cpu().numpy()
                perturbated_fake = IF_to_eeg(perturbated_fake, normalizer, iseeg=False, is_IF=True)[0].astype(
                    np.float64)
                perturbated_psd = mne.time_frequency.psd_array_welch(perturbated_fake, 64, n_fft=256)
                perturbated_psd_dist = np.linalg.norm(np.asarray(real_psd[0]) - np.asarray(perturbated_psd[0]),
                                                      ord=2) / len(real_psd[0]) ** 0.5
                psd_increment = perturbated_psd_dist - original_psd_dist
                # perturbated_dtw = dtw.distance_fast(real, perturbated_fake, use_pruning=True)
                # dtw_increment = perturbated_dtw - original_dtw
                mag_increment_ls.append(mag_increment)
                psd_increment_ls.append(psd_increment)
                # dtw_increment_ls.append(dtw_increment)

            corr1, p1 = pearsonr(mag_increment_ls, psd_increment_ls)
            # corr2, p2 = pearsonr(mag_increment_ls, dtw_increment_ls)
            correlations[band_name][EEG_chan].append(corr1)
            # correlations[band_name][EEG_chan][1].append(corr2)
            pvalue[band_name][EEG_chan].append(p1)
            # pvalue[band_name][EEG_chan][1].append(p2)

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... ' % i)

    np.save(os.path.join(saveDir, '{}_perturbationCorrPSD'.format(investigated)), correlations)
    np.save(os.path.join(saveDir, '{}_perturbationPvaluePSD'.format(investigated)), pvalue)


def add_right_cax(ax, pad, width):

    axpos = ax.get_position()
    caxpos = mpl.transforms.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0,
        axpos.x1 + pad + width,
        axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)

    return cax


def visualize_correlation_topomap(correlations, saveDir=None, investigated='all', seizure_set=[]):

    eeg_pos = np.load(EEGPosDir, allow_pickle=True).item()
    picked_eeg = {}
    display_chan = list(correlations['delta'].keys()) + seizure_set

    for chan in display_chan:
        picked_eeg[chan] = np.asarray(eeg_pos[chan]) / 1000. + np.asarray([0, 0.009, 0])
    for chan in eeg_pos.keys():
        if chan not in picked_eeg.keys():
            picked_eeg[chan] = np.asarray(eeg_pos[chan]) / 1000. + np.asarray([0, 0.009, 0])

    montage = mne.channels.make_dig_montage(picked_eeg)
    info = mne.create_info(ch_names=list(picked_eeg.keys()), ch_types=['eeg' for _ in range(len(picked_eeg))], sfreq=64.)
    info.set_montage(montage)

    n_eeg_placeholder = len(picked_eeg) - len(correlations['delta'])
    delta_corr = [np.mean(corr_ls) for chan, corr_ls in correlations['delta'].items()] + [0 for _ in range(n_eeg_placeholder)]
    theta_corr = [np.mean(corr_ls) for chan, corr_ls in correlations['theta'].items()] + [0 for _ in range(n_eeg_placeholder)]
    alpha_corr = [np.mean(corr_ls) for chan, corr_ls in correlations['alpha'].items()] + [0 for _ in range(n_eeg_placeholder)]
    beta_corr = [np.mean(corr_ls) for chan, corr_ls in correlations['beta'].items()] + [0 for _ in range(n_eeg_placeholder)]

    mask_list = np.zeros(len(picked_eeg)) != 0
    for i in range(len(display_chan) - len(seizure_set), len(display_chan)):
        mask_list[i] = True
    mask_params = dict(marker='o', markerfacecolor='red', markeredgecolor='yellow',
        linewidth=0, markersize=4)

    plt.rc('font', family='Times New Roman')
    fig = plt.figure()
    ax0 = fig.add_subplot(141)
    ax1 = fig.add_subplot(142)
    ax2 = fig.add_subplot(143)
    ax3 = fig.add_subplot(144)
    ax0.set_title('Delta', fontweight='bold', fontsize='xx-large')
    ax1.set_title('Theta', fontweight='bold', fontsize='xx-large')
    ax2.set_title('Alpha', fontweight='bold', fontsize='xx-large')
    ax3.set_title('Beta', fontweight='bold', fontsize='xx-large')
    im0, _ = mne.viz.plot_topomap(delta_corr, info, show=False, axes=ax0, cmap='RdBu_r', mask=mask_list, mask_params=mask_params, contours=0, names=list(correlations['delta'].keys()), show_names=True, vmin=0)  # , vmin=0, vmax=1
    im1, _ = mne.viz.plot_topomap(theta_corr, info, show=False, axes=ax1, cmap='RdBu_r', mask=mask_list, mask_params=mask_params, contours=0, names=list(correlations['delta'].keys()), show_names=True, vmin=0)  # names=info['ch_names']
    im2, _ = mne.viz.plot_topomap(alpha_corr, info, show=False, axes=ax2, cmap='RdBu_r', mask=mask_list, mask_params=mask_params, contours=0, names=list(correlations['delta'].keys()), show_names=True, vmin=0)
    im3, _ = mne.viz.plot_topomap(beta_corr, info, show=False, axes=ax3, cmap='RdBu_r', mask=mask_list, mask_params=mask_params, contours=0, names=list(correlations['delta'].keys()), show_names=True, vmin=0)
    
    cbar0 = fig.colorbar(im0, ax=ax0, shrink=0.2)
    cbar0.ax.set_ylabel("Correlation")
    # cax1 = add_right_cax(im1, pad=0.02, width=0.02)
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.2)
    cbar1.ax.set_ylabel("Correlation")
    # cax2 = add_right_cax(im2, pad=0.02, width=0.02)
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.2)
    cbar2.ax.set_ylabel("Correlation")
    # cax3 = add_right_cax(im3, pad=0.02, width=0.02)
    cbar3 = fig.colorbar(im3, ax=ax3, shrink=0.2)
    cbar3.ax.set_ylabel("Correlation")
    # cbar_ax = fig.add_axes([0.81, 0.35, 0.02, 0.33])
    # cbar = fig.colorbar(im3, cax=cbar_ax)
    # cbar.ax.set_ylabel("Correlation")

    # plt.title(' ', y=-0.3)
    plt.tight_layout(pad=0)
    if saveDir is not None:
        plt.savefig(os.path.join(saveDir, '{}_perturbationCorrelation.pdf'.format(investigated)))
    else:
        plt.show()


if __name__ == '__main__':

    topK = 100
    n_perturbation = 50
    investigated = 'lmk'
    aggregateLoc = False
    perturbation_saveDir = "/home/hmq/Infos/perturbation/"
    # perturbation_saveDir = '/public/home/xlwang/hmq/Infos/perturbation'

    correlations = np.load(os.path.join(perturbation_saveDir, '{}_perturbationCorrPSD.npy'.format(investigated)), allow_pickle=True).item()
    visualize_correlation_topomap(correlations, perturbation_saveDir, investigated, ['T4'])
