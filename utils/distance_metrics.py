import math
import torch
import scipy.optimize
from scipy.stats import pearsonr
from .eeg_tools import *
import random
from scipy.spatial.distance import pdist
from .numpy_tools import plot_hist_distribution
from dtaidistance import dtw
from scipy import signal
from numpy.lib.stride_tricks import as_strided
import cv2
from skimage import measure


band_width = 12
bands = ([0, 4], [5, 7], [8, 15], [16, 32], [0, 32])
conf = Configuration()


def get_bands():
    bands = []

    for i in range(0, conf.seeg_ceiling_freq, band_width):  
        j = i + band_width if i + band_width < conf.seeg_ceiling_freq else conf.seeg_ceiling_freq
        bands.append((i, j))
    return bands


def get_z_score_params(real_path, save_path=None):
    n_band = len(bands)
    dataset = make_dataset(real_path)
    v_sum = [0 for _ in range(n_band)]  
    stdv = [0 for _ in range(n_band)]
    num = [len(dataset) for _ in range(n_band)]   
    flag = True
    v = []
    z_score_params = np.load("/home/hmq/Infos/z_score_params/stft_Fz_224x224_ae_e2s_28Hz_z_score.npy", allow_pickle=True).item()  # depends on your data format
    t_sum = z_score_params['sum']
    t_mean = z_score_params['mean']
    t_stdv = z_score_params['stdv']

    for i, f_name in enumerate(dataset):

        f_n = os.path.basename(f_name).split('.')[0]
        real = np.load(os.path.join(real_path, f_n + '.npy'))[0]  

        for j, tup in enumerate(bands):

            _, r_mag, _ = get_time_freq_by_band(real, *tup, iseeg=False)
            if flag:
                num[j] *= r_mag.size
            v_sum[j] += np.sum(r_mag)
            if j == 4:
                rand = random.sample(range(r_mag.size), 1000)
                for r in rand:
                    row = r // r_mag.shape[1]
                    col = r % r_mag.shape[1]
                    v.append((r_mag[row][col] - t_mean[j]) / t_stdv[j])

        flag = False

    plot_hist_distribution(v, 14, 'Z-scored Magnitude', 'Frequency', title='0 to 28Hz', save_dir='/home/hmq/Projects/pix2pix/')
    return
    mean = [v_sum[i] / num[i] for i in range(n_band)]

    for i, f_name in enumerate(dataset):

        f_n = os.path.basename(f_name).split('.')[0]
        real = np.load(os.path.join(real_path, f_n + '.npy'))[0]  
        stdv[-1] += np.sum((real - mean[-1]) ** 2)

        for j, tup in enumerate(bands):

            _, r_mag, _ = get_time_freq_by_band(real, *tup, iseeg=False)
            stdv[j] += np.sum((r_mag - mean[j]) ** 2)

    stdv = [np.sqrt(stdv[i] / num[i]) for i in range(n_band)]

    if save_path is not None:
        np.save(save_path, {'sum': v_sum, 'mean': mean, 'stdv': stdv})

    return v_sum, mean, stdv


def shrinkage_estimator(weights, avg_dist, q=0.1):
    '''
    :param weights: np.ndarray, shape is (n_band, ), the weight of band, i.e. the sum of the energies in the band, normalized to a value between 0 and 1
    :param avg_dist: np.ndarray, shape is (n_band, 2), the mean of the difference between true and generated for each band, both L1 and L2 results are included
    :param q: a hyperparameter, default as 0.1
    
    :return: np.ndarray, shape is (n_band, 2), the average distance after shrinking
    '''

    weighted_avg_dist = np.average(avg_dist, weights=weights, axis=0)
    std = (np.linalg.norm(avg_dist - weighted_avg_dist, axis=0) / len(weights)) ** 0.5
    v_n = (1 + std / (q * np.abs(avg_dist - weighted_avg_dist))) ** (-1)
    shrunk_dist = v_n * weighted_avg_dist + (1 - v_n) * avg_dist

    return shrunk_dist


def pool2d(A, kernel_size, stride, padding, pool_mode='avg'):
    '''2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''

    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size,
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)


def cosine_distance(a, b):
    a_norm = np.linalg.norm(a.flatten(), ord=2)
    b_norm = np.linalg.norm(b.flatten(), ord=2)
    
    res = np.dot(a.flatten() / a_norm, b.flatten() / b_norm)
    return 1 - res


def hamming_distance(a, b):
    a_normed = (a - np.min(a)) / (np.max(a) - np.min(a)) * (a.size - 1)
    b_normed = (b - np.min(b)) / (np.max(b) - np.min(b)) * (b.size - 1)
    a_map = np.asarray(a_normed > np.average(a_normed), np.int32)
    b_map = np.asarray(b_normed > np.average(b_normed), np.int32)
    hamming_dist = pdist(np.vstack([a_map.flatten(), b_map.flatten()]), metric='hamming')[0]
    return hamming_dist


def ham_dist(x, y):
     """Get the hamming distance of two values.
        
     :param x:
     :param y:

     :return: the hamming distance
     """

     assert len(x) == len(y)
     return sum([ch1 != ch2 for ch1, ch2 in zip(x, y)])


def phash(img):
    # step1：resize the image to 32×32
    img=cv2.resize(img,(32,32))
    img=img.astype(np.float32)

    # step2: discrete cosine transformation
    img=cv2.dct(img)
    img=img[0:8,0:8]
    sum=0.
    hash_str=''

    # step3: calculate the average
    for i in range(8):
        for j in range(8):
            sum+=img[i,j]
    avg=sum/64

    # step4: get the hash
    for i in range(8):
        for j in range(8):
            if img[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str


def calculate_distance(real_path, fake_path, normalizer, z_score_path=None, method='GAN', is_IF=False, aggregate=True, save_path=None, aux_normalizer=None, baseline=None):
    '''

    :param aggregate: whether the results of all patients are aggregated, default as True
    :param real_path: the path of real seeg data
    :param fake_path: the path of generated seeg data
    :param normalizer: normalizer
    :param z_score_path: 
    :param method: str | 'EEG','GAN', 'EUD_temporal', 'EUD_freq', 'spline', 'asae'
    :param save_path: the path to save the result
    
    :return: result
    '''

    dataset = make_dataset(fake_path)
    if method in ['GAN', 'asae', 'ae', 'pghi', 'phase', 'EEGGAN', 'uTSGAN']:
        dataset = list(filter(lambda x: 'real_B' in x, dataset))
    
    n_band = len(bands)
    patient_result = {}

    for i, f_name in enumerate(dataset):

        print("iteration %d: " % i)

        f_n = os.path.basename(dataset[i])
        f_n = f_n.split('.')[0]
        patient = f_n.split('_')[0]
        if patient not in patient_result.keys():
            patient_result[patient] = {}
            patient_result[patient]['temporal_dist'] = []
            patient_result[patient]['rmse_dist'] = []
            patient_result[patient]['mag_dist'] = [[] for _ in range(n_band)]
            patient_result[patient]['psd_dist'] = [[] for _ in range(n_band)]
            patient_result[patient]['hd'] = []
            patient_result[patient]['bd'] = []
            patient_result[patient]['cosine'] = []
            patient_result[patient]['phase_dtw'] = []
            patient_result[patient]['phase_rmse'] = []
            patient_result[patient]['fake_mag_dtw'] = []
            patient_result[patient]['fake_mag_rmse'] = []
            patient_result[patient]['phase_dist'] = [[] for _ in range(n_band)]
            patient_result[patient]['phase_cosine'] = [[] for _ in range(n_band)]
            patient_result[patient]['IF_dist'] = [[] for _ in range(n_band)]
            patient_result[patient]['IF_cosine'] = [[] for _ in range(n_band)]
        f_n = '_'.join(f_n.split("_")[: 4])

        # read data according to method
        if method in ['GAN', 'phase', 'pghi']:
            fake = np.load(os.path.join(fake_path, f_n + '_fake_B.npy'))
        elif method in ['asae', 'ae', 'EEGGAN']:
            fake = np.load(os.path.join(fake_path, f_n + '_fake_seeg.npy'))
        elif method in ['EUD_temporal', 'EUD_freq', 'spline']:
            fake = np.load(os.path.join(fake_path, f_n.split('_')[1] + '.npy'))
        elif method in ['EEG']:
            fake = np.load(os.path.join(fake_path, f_n + '.npy'))[0]
        elif method in ['uTSGAN']:
            fake = np.load(os.path.join(fake_path, f_n + '_fake_B_temp.npy'))
        else:
            raise NotImplementedError

        # process inputs according to method
        if method in ['GAN', 'EUD_freq', 'ae']:
            fake = IF_to_eeg(fake, normalizer, iseeg=False, is_IF=is_IF)[0]
        elif method in ['asae', 'EEGGAN']:
            fake = normalizer.denormalize_temporal(fake, is_eeg=False)
        elif method in ['uTSGAN']:
            fake = normalizer.denormalize_temporal(fake, is_eeg=False)[0]

        real = np.load(os.path.join(real_path, patient, f_n + '.npy'))[0]  
        if len(real) > fake.shape[0]:
            real = real[: fake.shape[0]]
        r_temp, r_mag, _, _ = get_time_freq_by_band(real, 0, conf.seeg_ceiling_freq, iseeg=False)  

        fake = fake.astype(np.float64)  # transfer original signal to float64 for the fake signal is generated float32

        if method in ['EEG', 'EUD_temporal', 'spline', 'asae', 'EEGGAN']:
            fake, _, _, _ = get_time_freq_by_band(fake, 0, conf.seeg_ceiling_freq, iseeg=False)

        dtw_dist = dtw.distance_fast(r_temp, fake, use_pruning=True)
        rmse_dist = np.linalg.norm(r_temp - fake, ord=2) / len(r_temp) ** 0.5
        patient_result[patient]['temporal_dist'].append(dtw_dist)
        patient_result[patient]['rmse_dist'].append(rmse_dist)

        for j, tup in enumerate(bands):

            _, r_mag, r_p, r_IF = get_time_freq_by_band(real, *tup, iseeg=False)
            _, f_mag, f_p, f_IF = get_time_freq_by_band(fake, *tup, iseeg=False)
            
            # calculate hamming distance
            hamming_dist = ham_dist(phash(r_mag), phash(f_mag))
            patient_result[patient]['mag_dist'][j].append(hamming_dist)

            # calculate psd distance
            r_psd = mne.time_frequency.psd_array_welch(real, conf.seeg_sf, fmin=tup[0], fmax=tup[1], n_fft=conf.seeg_n_fft)
            f_psd = mne.time_frequency.psd_array_welch(fake, conf.seeg_sf, fmin=tup[0], fmax=tup[1], n_fft=conf.seeg_n_fft)
            dist = np.linalg.norm(np.asarray(r_psd[0]) - np.asarray(f_psd[0]), ord=2) / len(r_psd[0]) ** 0.5
            patient_result[patient]['psd_dist'][j].append(dist)

            # calculate phase distance
            patient_result[patient]['phase_dist'][j].append(ham_dist(phash(r_p), phash(f_p)))
            patient_result[patient]['phase_cosine'][j].append(cosine_distance(r_p, f_p))
            patient_result[patient]['IF_dist'][j].append(ham_dist(phash(r_IF), phash(f_IF)))
            patient_result[patient]['IF_cosine'][j].append(cosine_distance(r_IF, f_IF))

            # calculate the hamming distance of phase/IF
            if j == len(bands) - 1:
                r_psd_prob = psd_probability(r_psd[0], conf.seeg_n_fft, real.shape[0])
                f_psd_prob = psd_probability(f_psd[0], conf.seeg_n_fft, real.shape[0])
                
                hd = Hellinger_Distance(r_psd_prob, f_psd_prob)
                bd = Bhattacharyya_Distance(r_psd_prob, f_psd_prob)
                patient_result[patient]['hd'].append(hd)
                patient_result[patient]['bd'].append(bd)
                
                # calculate cosine distance
                patient_result[patient]['cosine'].append(cosine_distance(r_mag, f_mag))
                
                # calculate the results after replacing the phase
                if is_IF:
                    phase_fake = np.stack((np.log(r_mag + conf.epsilon), f_IF), axis=0)
                else:
                    phase_fake = np.stack((np.log(r_mag + conf.epsilon), f_p), axis=0)
                
                phase_fake = aux_normalizer.normalize(torch.from_numpy(phase_fake[np.newaxis, ...]), 'seeg').numpy()[0]
                phase_fake = IF_to_eeg(phase_fake, aux_normalizer, iseeg=False, is_IF=is_IF)[0]
                phase_dtw_dist = dtw.distance_fast(r_temp, phase_fake, use_pruning=True)
                phase_rmse_dist = np.linalg.norm(r_temp - phase_fake, ord=2) / len(r_temp) ** 0.5
                patient_result[patient]['phase_dtw'].append(phase_dtw_dist)
                patient_result[patient]['phase_rmse'].append(phase_rmse_dist)
                
                # calculate the results after replacing the mag
                if is_IF:
                    mag_fake = np.stack((np.log(f_mag + conf.epsilon), r_IF), axis=0)
                else:
                    mag_fake = np.stack((np.log(f_mag + conf.epsilon), r_p), axis=0)
                
                mag_fake = aux_normalizer.normalize(torch.from_numpy(mag_fake[np.newaxis, ...]), 'seeg').numpy()[0]
                mag_fake = IF_to_eeg(mag_fake, aux_normalizer, iseeg=False, is_IF=is_IF)[0]
                mag_fake_dtw_dist = dtw.distance_fast(r_temp, mag_fake, use_pruning=True)
                mag_fake_rmse_dist = np.linalg.norm(r_temp - mag_fake, ord=2) / len(r_temp) ** 0.5
                patient_result[patient]['fake_mag_dtw'].append(mag_fake_dtw_dist)
                patient_result[patient]['fake_mag_rmse'].append(mag_fake_rmse_dist)

        print('Temporal:')
        print(dtw_dist)

    result = None
    
    if aggregate:

        temporal_array = []
        rmse_array = []
        bd_array = []
        hd_array = []
        cos_array = []
        phase_dtw_array = []
        phase_rmse_array = []
        mag_fake_dtw_array = []
        mag_fake_rmse_array = []
        mag_array = [[] for _ in range(n_band)]
        psd_array = [[] for _ in range(n_band)]
        phase_dist_array = [[] for _ in range(n_band)]
        phase_cos_array = [[] for _ in range(n_band)]
        IF_dist_array = [[] for _ in range(n_band)]
        IF_cos_array = [[] for _ in range(n_band)]

        for _, d in patient_result.items():
            temporal_array += d['temporal_dist']
            rmse_array += d['rmse_dist']
            bd_array += d['bd']
            hd_array += d['hd']
            cos_array += d['cosine']
            phase_dtw_array += d['phase_dtw']
            phase_rmse_array += d['phase_rmse']
            mag_fake_dtw_array += d['fake_mag_dtw']
            mag_fake_rmse_array += d['fake_mag_rmse']
            for i in range(n_band):
                mag_array[i].extend(d['mag_dist'][i])
                psd_array[i].extend(d['psd_dist'][i])
                phase_dist_array[i].extend(d['phase_dist'][i])
                phase_cos_array[i].extend(d['phase_cosine'][i])
                IF_dist_array[i].extend(d['IF_dist'][i])
                IF_cos_array[i].extend(d['IF_cosine'][i])

        if baseline is not None:
            temporal_array = scale_results(temporal_array, baseline['temporal_mean'])
            psd_array = scale_results(psd_array, baseline['psd_mean'][-1])  
            hd_array = scale_results(hd_array, baseline['hd_mean'])
            mag_fake_dtw_array = scale_results(mag_fake_dtw_array, baseline['temporal_mean'])

        temporal_mean = np.mean(temporal_array)
        temporal_std = np.std(temporal_array, ddof=1)
        mag_mean = np.mean(mag_array, axis=1)
        mag_std = np.std(mag_array, axis=1, ddof=1)
        psd_mean = np.mean(psd_array, axis=1)
        psd_std = np.std(psd_array, axis=1, ddof=1)
        rmse_mean = np.mean(rmse_array)
        rmse_std = np.std(rmse_array, ddof=1)
        bd_mean = np.mean(bd_array)
        bd_std = np.std(bd_array, ddof=1)
        hd_mean = np.mean(hd_array)
        hd_std = np.std(hd_array, ddof=1)
        cos_mean = np.mean(cos_array)
        cos_std = np.std(cos_array, ddof=1)
        phase_dtw_mean = np.mean(phase_dtw_array)
        phase_dtw_std = np.std(phase_dtw_array, ddof=1)
        phase_rmse_mean = np.mean(phase_rmse_array)
        phase_rmse_std = np.std(phase_rmse_array, ddof=1)
        mag_fake_dtw_mean = np.mean(mag_fake_dtw_array)
        mag_fake_dtw_std = np.std(mag_fake_dtw_array, ddof=1)
        mag_fake_rmse_mean = np.mean(mag_fake_rmse_array)
        mag_fake_rmse_std = np.std(mag_fake_rmse_array, ddof=1)
        phase_dist_mean = np.mean(phase_dist_array, axis=1)
        phase_dist_std = np.std(phase_dist_array, axis=1, ddof=1)
        phase_cos_mean = np.mean(phase_cos_array, axis=1)
        phase_cos_std = np.std(phase_cos_array, axis=1, ddof=1)
        IF_dist_mean = np.mean(IF_dist_array, axis=1)
        IF_dist_std = np.std(IF_dist_array, axis=1, ddof=1)
        IF_cos_mean = np.mean(IF_cos_array, axis=1)
        IF_cos_std = np.std(IF_cos_array, axis=1, ddof=1)

        result = {'temporal_mean': temporal_mean, 'temporal_std': temporal_std,
                  'mag_mean': mag_mean, 'mag_std': mag_std, 'psd_mean': psd_mean, 'psd_std': psd_std,
                  'rmse_mean': rmse_mean, 'rmse_std': rmse_std,
                  'bd_mean': bd_mean, 'bd_std':bd_std, 'hd_mean': hd_mean, 'hd_std': hd_std, 'cos_mean': cos_mean,
                  'cos_std': cos_std, 'phase_dtw_mean': phase_dtw_mean, 'phase_dtw_std': phase_dtw_std,
                  'phase_rmse_mean': phase_rmse_mean, 'phase_rmse_std': phase_rmse_std,
                  'fake_mag_dtw_mean': mag_fake_dtw_mean, 'fake_mag_dtw_std': mag_fake_dtw_std,
                  'fake_mag_rmse_mean': mag_fake_rmse_mean, 'fake_mag_rmse_std': mag_fake_rmse_std,
                  'phase_dist_mean': phase_dist_mean, 'phase_dist_std': phase_dist_std,
                  'phase_cos_mean': phase_cos_mean, 'phase_cos_std': phase_cos_std,
                  'IF_dist_mean': IF_dist_mean, 'IF_dist_std': IF_dist_std,
                  'IF_cos_mean': IF_cos_mean, 'IF_cos_std': IF_cos_std
                  }

    else:
        result = {}
        for p, d in patient_result.items():

            if baseline is not None:
                d['temporal_dist'] = scale_results(d['temporal_dist'], baseline['temporal_mean'])
                d['psd_dist'] = scale_results(d['psd_dist'], baseline['psd_mean'][-1])  
                d['hd'] = scale_results(d['hd'], baseline['hd_mean'])
                d['fake_mag_dtw'] = scale_results(d['fake_mag_dtw'], baseline['temporal_mean'])

            temporal_mean = np.mean(d['temporal_dist'])
            temporal_std = np.std(d['temporal_dist'], ddof=1)
            mag_mean = np.mean(d['mag_dist'], axis=1)
            mag_std = np.std(d['mag_dist'], axis=1, ddof=1)
            psd_mean = np.mean(d['psd_dist'], axis=1)
            psd_std = np.std(d['psd_dist'], axis=1, ddof=1)
            rmse_mean = np.mean(d['rmse_dist'])
            rmse_std = np.std(d['rmse_dist'], ddof=1)
            bd_mean = np.mean(d['bd'])
            bd_std = np.std(d['bd'], ddof=1)
            hd_mean = np.mean(d['hd'])
            hd_std = np.std(d['hd'], ddof=1)
            cos_mean = np.mean(d['cosine'])
            cos_std = np.std(d['cosine'], ddof=1)
            phase_dtw_mean = np.mean(d['phase_dtw'])
            phase_dtw_std = np.std(d['phase_dtw'], ddof=1)
            phase_rmse_mean = np.mean(d['phase_rmse'])
            phase_rmse_std = np.std(d['phase_rmse'], ddof=1)
            mag_fake_dtw_mean = np.mean(d['fake_mag_dtw'])
            mag_fake_dtw_std = np.std(d['fake_mag_dtw'], ddof=1)
            mag_fake_rmse_mean = np.mean(d['fake_mag_rmse'])
            mag_fake_rmse_std = np.std(d['fake_mag_rmse'], ddof=1)
            phase_dist_mean = np.mean(d['phase_dist'], axis=1)
            phase_dist_std = np.std(d['phase_dist'], axis=1, ddof=1)
            phase_cos_mean = np.mean(d['phase_cosine'], axis=1)
            phase_cos_std = np.std(d['phase_cosine'], axis=1, ddof=1)
            IF_dist_mean = np.mean(d['IF_dist'], axis=1)
            IF_dist_std = np.std(d['IF_dist'], axis=1, ddof=1)
            IF_cos_mean = np.mean(d['IF_cosine'], axis=1)
            IF_cos_std = np.std(d['IF_cosine'], axis=1, ddof=1)

            result[p] = {'temporal_mean': temporal_mean, 'temporal_std': temporal_std,
                         'mag_mean': mag_mean, 'mag_std': mag_std, 'psd_mean': psd_mean, 'psd_std': psd_std,
                         'rmse_mean': rmse_mean, 'rmse_std': rmse_std,
                         'bd_mean': bd_mean, 'bd_std': bd_std, 'hd_mean': hd_mean, 'hd_std': hd_std,
                         'cos_mean': cos_mean, 'cos_std': cos_std, 'phase_dtw_mean': phase_dtw_mean, 'phase_dtw_std': phase_dtw_std,
                         'phase_rmse_mean': phase_rmse_mean, 'phase_rmse_std': phase_rmse_std,
                         'fake_mag_dtw_mean': mag_fake_dtw_mean, 'fake_mag_dtw_std': mag_fake_dtw_std,
                         'fake_mag_rmse_mean': mag_fake_rmse_mean, 'fake_mag_rmse_std': mag_fake_rmse_std,
                         'phase_dist_mean': phase_dist_mean, 'phase_dist_std': phase_dist_std,
                         'phase_cos_mean': phase_cos_mean, 'phase_cos_std': phase_cos_std,
                         'IF_dist_mean': IF_dist_mean, 'IF_dist_std': IF_dist_std,
                         'IF_cos_mean': IF_cos_mean, 'IF_cos_std': IF_cos_std
                         }

    if save_path is not None:
        np.save(save_path, result)

    return result


def scale_results(metric_array, baseline):

    metric_array = np.asarray(metric_array)
    metric_array /= baseline
    metric_array = np.log2(metric_array)

    return metric_array


def get_neighbors_distance(neighbor_pos, target_pos, nearest_k=None, method='EUD'):
    ''' Get the k nearest neighbor channels of the current channel, and the distance calculated by the method.
    
    :param neighbor_pos: dict, the coordinates of neighbor channels
    :param target_pos: array like, (3, ), the coordinate of current channel
    :param nearest_k: int, the number of nearest channels to be included (include current channel)
    :param method: str, the method to calculate the distance
    
    :return: the name and distance of the neighbors
    '''
    
    target_pos = np.asarray(target_pos)
    pos = np.asarray(list(neighbor_pos.values()))
    ch_names = list(neighbor_pos.keys())

    if method == 'EUD':
        dist = np.linalg.norm(pos - target_pos, axis=1)
        if nearest_k is None:
            neighbors = sorted(zip(dist, ch_names))
        else:
            neighbors = sorted(zip(dist, ch_names))[:nearest_k]
        neighbor_names = [tup[1] for tup in neighbors]
        neighbor_dist = np.asarray([tup[0] for tup in neighbors])
        return neighbor_names, neighbor_dist

    elif method == 'EGL':
        pass


def psd_probability(psd, n_fft, data_len):

    window = signal.tukey(n_fft)

    M = n_fft
    L = data_len // M
    U = np.sum(window ** 2) / M

    return psd / (M * U * L)


def Hellinger_Distance(array_1, array_2):
    '''

    :param array_1: ndarray (, n)
    :param array_2: ndarray (, n)
    
    :return: 
    '''

    if isinstance(array_1, list):
        array_1 = np.asarray(array_1)
    if isinstance(array_2, list):
        array_2 = np.asarray(array_2)
    if array_1.shape != array_2.shape:
        print('Error! Array not aligned!')
        return

    h_d = np.linalg.norm(array_1 ** 0.5 - array_2 ** 0.5, ord=2) / 2 ** 0.5
    similarity = h_d ** -1
    return h_d


def Bhattacharyya_Distance(array_1, array_2):
    '''
    
    :param array_1: ndarray (, n)
    :param array_2: ndarray (, n)
    
    :return: 
    '''

    if isinstance(array_1, list):
        array_1 = np.asarray(array_1)
    if isinstance(array_2, list):
        array_2 = np.asarray(array_2)
    if array_1.shape != array_2.shape:
        print('Error! Array not aligned!')
        return
    bc_coef = np.sum((array_1 * array_2) ** 0.5)
    similarity = -np.log(bc_coef)
    return similarity ** -1


def EUD_interpolate(segment, dist, save_dir=None, save_name=None):
    '''

    :param segment: a segment with multiple channels
    :param dir: the directory of the segment
    :param neighbors: the name of neighbor channels
    :param dist: distance
    
    :return:
    '''

    feat_shape = segment[0].shape
    neighbors = segment.reshape((segment.shape[0], -1)).T
    interpolated = np.zeros(feat_shape).ravel()

    for i, t in enumerate(neighbors):
        interpolated[i] = np.sum(t / dist) / np.sum(dist ** -1)

    interpolated = interpolated.reshape(feat_shape)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, save_name), interpolated)

    return interpolated


def spline_interpolate(raw, pos_dict, seeg_name):
    ''' Use the SPLINE interpolation in mne library to generate SEEG.
    
    :param raw: mne.io.Raw
    :param pos_dict: a dictionary includes names and coordinates of eeg and seeg channels to be interpolated
    :param seeg_name: the name of target seeg channel
    
    :return: mne.io.Raw, result after interpolation
    '''
    
    raw.pick_channels(list(pos_dict.keys()))
    raw.set_channel_types(dict([(k, 'eeg') for k in list(pos_dict.keys())]))
    montage = mne.channels.make_dig_montage(pos_dict)
    raw.set_montage(montage)
    raw.info['bads'].append(seeg_name)
    raw.interpolate_bads(method=dict(eeg='spline'), origin=(0., 0., 0.))
    raw.pick_channels([seeg_name])

    return raw


def explore_eeg_seeg_dist_sim_relation(patient, distFile, bdSimFile, hdSimFile, saveDir=None):
    '''Plot the distance vs. similarity between each EEG channel and all SEEG channels for each patient (scatter plot & straight line fitting)
    
    :param patient: str, the name of a patient
    :param distFile: str, the path of coordinate distance file
    :param bdSimFile: str, the path of bd similarity file
    :param hdSimFile: str, the path of hd similarity file
    :param saveDir: str, the directory to be saved
    
    :return: (float, float), the proportion of the slope of the fitting line is negative under the two similarities
    '''

    n_sample = 5
    distDict = np.load(distFile, allow_pickle=True).item()
    bdSimDict = np.load(bdSimFile, allow_pickle=True).item()
    hdSimDict = np.load(hdSimFile, allow_pickle=True).item()
    
    eegChans = distDict.keys()
    bdNegSlope = 0
    hdNegSlope = 0
    intvLen = 160
    distIntv = list(range(0, 161, intvLen))

    for i, chan in enumerate(eegChans):
        fullX = np.asarray([float(tup[1]) for tup in distDict[chan]])
        fullY1 = []
        fullY2 = []

        bdDict = {k: v for (k, v) in bdSimDict[chan]}
        hdDict = {k: v for (k, v) in hdSimDict[chan]}

        for tup in distDict[chan]:
            fullY1.append(float(bdDict[tup[0]]))
            fullY2.append(float(hdDict[tup[0]]))
        fullY1 = np.asarray(fullY1)
        fullY2 = np.asarray(fullY2)

        '''plot bd'''
        plt.scatter(fullX, fullY1, c='red')
        for intv_idx in range(len(distIntv) - 1):
            span = np.where((fullX > distIntv[intv_idx]) & (fullX <= distIntv[intv_idx + 1]), True,
                            False)  
            if np.any(span):
                X = fullX[span]
                Y1 = fullY1[span]
                A1, B1 = scipy.optimize.curve_fit(lambda x, A, B: A*x + B, X, Y1)[0]
                X1 = np.arange(distIntv[intv_idx], distIntv[intv_idx + 1], 1)
                Y11 = A1 * X1 + B1
                
                plt.plot(X1, Y11, 'blue')
            
            else:
                plt.plot([distIntv[intv_idx]], min(fullY1), 'blue')
        
        plt.title(' '.join([patient, chan, str(intvLen)]))
        
        plt.xlabel('SEEG-EEG Electroid Physical Distance')
        plt.ylabel('Bhattacharyya Distance')

        if saveDir is None:
            plt.show()
        else:
            plt.savefig(os.path.join(saveDir, '_'.join([patient, chan, 'bd', str(intvLen)])))

        plt.close()


        '''plot hd'''
        plt.scatter(fullX, fullY2, c='red')
        for intv_idx in range(len(distIntv) - 1):
            span = np.where((fullX > distIntv[intv_idx]) & (fullX <= distIntv[intv_idx + 1]), True,
                            False)  
            if np.any(span):
                X = fullX[span]
                Y2 = fullY2[span]
                X1 = np.arange(distIntv[intv_idx], distIntv[intv_idx + 1], 1)
                A2, B2 = scipy.optimize.curve_fit(lambda x, A, B: A * x + B, X, Y2)[0]
                Y22 = A2 * X1 + B2
                
                plt.plot(X1, Y22, 'blue')
            
            else:
                plt.plot([distIntv[intv_idx]], min(fullY2), 'blue')
        
        plt.title(' '.join([patient, chan, str(intvLen)]))
        plt.legend()
        plt.xlabel('SEEG-EEG Electroid Physical Distance')
        plt.ylabel('Hellinger Distance')

        if saveDir is None:
            plt.show()
        else:
            plt.savefig(os.path.join(saveDir, '_'.join([patient, chan, 'hd', str(intvLen)])))

        plt.close()


def explore_patient_dist_sim_relation(patient, distFile, bdSimFile, hdSimFile, saveDir=None):
    ''' Plot the distance vs. similarity between all EEG channels and all SEEG channels for each patient (scatter plot & straight line fitting)
    
    :param patient: str, the name of a patient
    :param distFile: str, the path of coordinate distance file
    :param bdSimFile: str, the path of bd similarity file
    :param hdSimFile: str, the path of hd similarity file
    :param saveDir: str, the directory to be saved
    
    :return: 
    '''

    distDict = np.load(distFile, allow_pickle=True).item()
    bdSimDict = np.load(bdSimFile, allow_pickle=True).item()
    hdSimDict = np.load(hdSimFile, allow_pickle=True).item()
    X = []
    Y1 = []
    Y2 = []

    for chan in distDict.keys():

        X.extend([float(tup[1]) for tup in distDict[chan]])
        bdDict = {k: v for (k, v) in bdSimDict[chan]}
        hdDict = {k: v for (k, v) in hdSimDict[chan]}

        for tup in distDict[chan]:
            Y1.append(float(bdDict[tup[0]]))
            Y2.append(float(hdDict[tup[0]]))

    '''plot bd'''
    corr, p = scipy.stats.pearsonr(X, Y1)
    plt.scatter(X, Y1, c='red', label='corr={}, p={}'.format(np.around(corr, 4), np.around(p, 4)))
    A1, B1 = scipy.optimize.curve_fit(lambda x, A, B: A*x + B, X, Y1)[0]
    X1 = np.arange(math.floor(min(X)), math.ceil(max(X)), 1)
    Y11 = A1 * X1 + B1
    plt.plot(X1, Y11, 'blue', label='y={}x+{}'.format(np.around(A1, 4), np.around(B1, 4)))
    plt.title(' '.join([patient, 'all']))
    plt.legend()
    plt.xlabel('SEEG-EEG Electroid Physical Distance')
    plt.ylabel('Bhattacharyya Distance')

    if saveDir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(saveDir, '_'.join([patient, 'all', 'bd'])))

    plt.close()

    '''plot hd'''
    corr, p = scipy.stats.pearsonr(X, Y2)
    plt.scatter(X, Y2, c='red', label='corr={}, p={}'.format(np.around(corr, 4), np.around(p, 4)))
    A2, B2 = scipy.optimize.curve_fit(lambda x, A, B: A * x + B, X, Y2)[0]
    Y22 = A2 * X1 + B2
    plt.plot(X1, Y22, 'blue', label='y=' + str(np.around(A2, 4)) + 'x+' + str(np.around(B2, 4)))
    plt.title(' '.join([patient, 'all']))
    plt.legend()
    plt.xlabel('SEEG-EEG Electroid Physical Distance')
    plt.ylabel('Hellinger Distance')

    if saveDir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(saveDir, '_'.join([patient, 'all', 'hd'])))

    plt.close()


def explore_segment_dist_sim_relation(patient, distFile, saveDir=None):
    ''' Plot the distance vs. similarity between each EEG channel and all SEEG channels for each patient, by evenly sampling a time slice of 28 sec length (scatter plot & straight line fitting)
    
    :param patient: str, the name of a patient
    :param distFile: str, the path of coordinate distance file
    :param saveDir: str, the directory to be saved
    
    :return:
    '''

    rawEEG = read_raw_signal(os.path.join("/home/hmq/Signal/preprocessed/", patient + '_eeg_raw.fif'))  # depends on your data
    rawSEEG = read_raw_signal(os.path.join("/home/hmq/Signal/preprocessed/", patient + '_seeg_raw.fif'))
    distDict = np.load(distFile, allow_pickle=True).item()
    n_segment = 100
    duration = 28
    startPoints = random.sample(range(int(rawEEG.times.max()) - 30), n_segment)
    bdNegSlope = [0 for _ in range(len(distDict.keys()))]
    hdNegSlope = [0 for _ in range(len(distDict.keys()))]
    results = []
    bdmaxAllTime = {}
    hdmaxAllTime = {}
    bdmaxAllTime['R2'] = float('-inf')
    hdmaxAllTime['R2'] = float('-inf')

    for i, chan in enumerate(distDict.keys()):

        bdmaxR2 = {'R2': float('-inf')}
        hdmaxR2 = {'R2': float('-inf')}
        seegOrder = [tup[0] for tup in distDict[chan]]
        X = [float(tup[1]) for tup in distDict[chan]]

        pickedEEG = rawEEG.copy().pick_channels([chan])
        rawSEEG.reorder_channels(seegOrder)

        for start in startPoints:

            eegPSD = mne.time_frequency.psd_welch(pickedEEG, tmin=start, tmax=start + duration, fmin=0, fmax=28, n_fft=conf.eeg_n_fft)[0][0]
            seegPSD = mne.time_frequency.psd_welch(rawSEEG, tmin=start, tmax=start + duration, fmin=0, fmax=28, n_fft=conf.seeg_n_fft)[0]
            Y1 = [Bhattacharyya_Distance(eegPSD, sp) for sp in seegPSD]
            Y2 = [Hellinger_Distance(eegPSD, sp) for sp in seegPSD]

            corr1, p1 = scipy.stats.pearsonr(X, Y1)
            A1, B1 = scipy.optimize.curve_fit(lambda x, A, B: A * x + B, X, Y1)[0]
            X1 = np.arange(math.floor(min(X)), math.ceil(max(X)), 1)
            Y11 = A1 * X1 + B1
            r21 = 1 - np.mean((np.asarray(Y1) - A1 * np.asarray(X) - B1) ** 2) / np.var(Y1)
            bdNegSlope[i] = bdNegSlope[i] + 1 if A1 < 0 else bdNegSlope[i]

            corr2, p2 = scipy.stats.pearsonr(X, Y2)
            A2, B2 = scipy.optimize.curve_fit(lambda x, A, B: A * x + B, X, Y2)[0]
            X2 = np.arange(math.floor(min(X)), math.ceil(max(X)), 1)
            Y21 = A2 * X2 + B2
            r22 = 1 - np.mean((np.asarray(Y2) - A2 * np.asarray(X) - B2) ** 2) / np.var(Y2)
            hdNegSlope[i] = hdNegSlope[i] + 1 if A2 < 0 else hdNegSlope[i]

            if r21 > bdmaxR2['R2']:
                bdmaxR2['R2'] = r21
                bdmaxR2['eeg'] = chan
                bdmaxR2['start'] = start
                bdmaxR2['X'] = X
                bdmaxR2['Y'] = Y1
                bdmaxR2['A'] = A1
                bdmaxR2['B'] = B1
                bdmaxR2['Yline'] = Y11
                bdmaxR2['corr'] = corr1
                bdmaxR2['p'] = p1
            if r22 > hdmaxR2['R2']:
                hdmaxR2['R2'] = r22
                hdmaxR2['eeg'] = chan
                hdmaxR2['start'] = start
                hdmaxR2['X'] = X
                hdmaxR2['Y'] = Y2
                hdmaxR2['A'] = A2
                hdmaxR2['B'] = B2
                hdmaxR2['Yline'] = Y21
                hdmaxR2['corr'] = corr2
                hdmaxR2['p'] = p2

            results.append({'eeg': chan, 'start': start, 'corr_bd': corr1, 'p_bd': p1, 'A_bd': A1, 'B_bd': B1, 'R2_bd': r21,
                            'corr_hd': corr2, 'p_hd': p2, 'A_hd': A2, 'B_hd': B2, 'R2_hd': r22})

        del pickedEEG

        if bdmaxR2['R2'] > bdmaxAllTime['R2']:
            bdmaxAllTime.update(bdmaxR2)
        if hdmaxR2['R2'] > hdmaxAllTime['R2']:
            hdmaxAllTime.update(hdmaxR2)

        '''plot bd'''
        plt.scatter(bdmaxR2['X'], bdmaxR2['Y'], c='red',
                    label='corr={}, p={}'.format(np.around(bdmaxR2['corr'], 4), np.around(bdmaxR2['p'], 4)))
        X1 = np.arange(math.floor(min(bdmaxR2['X'])), math.ceil(max(bdmaxR2['X'])), 1)
        plt.plot(X1, bdmaxR2['Yline'], 'blue',
                 label='y={}x+{}'.format(np.around(bdmaxR2['A'], 4), np.around(bdmaxR2['B'], 4)))
        plt.title(' '.join([patient, bdmaxR2['eeg']]))
        plt.legend()
        plt.xlabel('SEEG-EEG Electroid Physical Distance')
        plt.ylabel('Bhattacharyya Similarity')

        if saveDir is None:
            plt.show()
        else:
            plt.savefig(os.path.join(saveDir, '_'.join([patient, 'segment', chan, 'bd'])))

        plt.close()

        '''plot hd'''
        plt.scatter(hdmaxR2['X'], hdmaxR2['Y'], c='red', label='corr={}, p={}'.format(np.around(hdmaxR2['corr'], 4), np.around(hdmaxR2['p'], 4)))
        X1 = np.arange(math.floor(min(hdmaxR2['X'])), math.ceil(max(hdmaxR2['X'])), 1)
        plt.plot(X1, hdmaxR2['Yline'], 'blue', label='y={}x+{}'.format(np.around(hdmaxR2['A'], 4), np.around(hdmaxR2['B'], 4)))
        plt.title(' '.join([patient, hdmaxR2['eeg']]))
        plt.legend()
        plt.xlabel('SEEG-EEG Electroid Physical Distance')
        plt.ylabel('Hellinger Similarity')

        if saveDir is None:
            plt.show()
        else:
            plt.savefig(os.path.join(saveDir, '_'.join([patient, 'segment', chan, 'hd'])))

        plt.close()

    bdNegRate = [v * 1. / n_segment for v in bdNegSlope]
    hdNegRate = [v * 1. / n_segment for v in hdNegSlope]

    return bdmaxAllTime, hdmaxAllTime, bdNegRate, hdNegRate, np.mean(bdNegRate), np.mean(hdNegRate), results


def explore_segment_range_dist_sim_relation(patient, distFile, intvLen, saveDir=None):
    ''' Plot the distance vs. similarity between each EEG channel and all SEEG channels for each patient, according to the distance range,
    by evenly sampling a time slice of 28 sec length (scatter plot & straight line fitting)
    
    :param patient: str, the name of a patient
    :param distFile: str, the path of coordinate distance file
    :param intvLen: int, the length of distance interval
    :param saveDir: str, the directory to be saved
    
    :return:
    '''

    rawEEG = read_raw_signal(os.path.join("/home/hmq/Signal/preprocessed/", patient + '_eeg_raw.fif'))  # depends on your data
    rawSEEG = read_raw_signal(os.path.join("/home/hmq/Signal/preprocessed/", patient + '_seeg_raw.fif'))
    distDict = np.load(distFile, allow_pickle=True).item()
    n_segment = 100
    duration = 28
    n_chan = 1
    n_exper = 20
    maxDist = 181
    bdNegSlope = np.zeros((n_chan, maxDist // intvLen, n_exper))
    hdNegSlope = np.zeros((n_chan, maxDist // intvLen, n_exper))
    results = []
    bdmaxAllTime = {}
    hdmaxAllTime = {}
    bdmaxAllTime['R2'] = float('-inf')
    hdmaxAllTime['R2'] = float('-inf')
    distIntv = list(range(0, maxDist, intvLen))
    nSEEGperIntv = np.zeros((n_chan, maxDist // intvLen))

    for i, chan in enumerate(distDict.keys()):

        if chan != 'CZ':
            continue
        else:
            i = 0

        bdmaxR2 = {'R2': float('-inf')}
        hdmaxR2 = {'R2': float('-inf')}
        seegOrder = [tup[0] for tup in distDict[chan]]
        fullX = np.asarray([float(tup[1]) for tup in distDict[chan]])

        pickedEEG = rawEEG.copy().pick_channels([chan])
        rawSEEG.reorder_channels(seegOrder)

        for expr_idx in range(n_exper):
            startPoints = random.sample(range(int(rawEEG.times.max()) - 30), n_segment)
            for start in startPoints:

                eegPSD = mne.time_frequency.psd_welch(pickedEEG, tmin=start, tmax=start + duration, fmin=0, fmax=28, n_fft=conf.eeg_n_fft)[0][0]
                seegPSD = mne.time_frequency.psd_welch(rawSEEG, tmin=start, tmax=start + duration, fmin=0, fmax=28, n_fft=conf.seeg_n_fft)[0]
                fullY1 = np.asarray([Bhattacharyya_Distance(eegPSD, sp) for sp in seegPSD])
                fullY2 = np.asarray([Hellinger_Distance(eegPSD, sp) for sp in seegPSD])

                for intv_idx in range(len(distIntv) - 1):
                    span = np.where((fullX > distIntv[intv_idx]) & (fullX <= distIntv[intv_idx + 1]), True, False)

                    if np.any(span):
                        X = fullX[span]
                        Y1 = fullY1[span]
                        nSEEGperIntv[i][intv_idx] = len(X)

                        corr1, p1 = scipy.stats.pearsonr(X, Y1)
                        A1, B1 = scipy.optimize.curve_fit(lambda x, A, B: A * x + B, X, Y1)[0]
                        X1 = np.arange(distIntv[intv_idx], distIntv[intv_idx + 1], 1)
                        Y11 = A1 * X1 + B1
                        r21 = 1 - np.mean((np.asarray(Y1) - A1 * np.asarray(X) - B1) ** 2) / np.var(Y1)
                        bdNegSlope[i][intv_idx][expr_idx] = bdNegSlope[i][intv_idx][expr_idx] + 1 if A1 < 0 else bdNegSlope[i][intv_idx][expr_idx]

                        Y2 = fullY2[span]
                        corr2, p2 = scipy.stats.pearsonr(X, Y2)
                        A2, B2 = scipy.optimize.curve_fit(lambda x, A, B: A * x + B, X, Y2)[0]
                        X2 = np.arange(distIntv[intv_idx], distIntv[intv_idx + 1], 1)
                        Y21 = A2 * X2 + B2
                        r22 = 1 - np.mean((np.asarray(Y2) - A2 * np.asarray(X) - B2) ** 2) / np.var(Y2)
                        hdNegSlope[i][intv_idx][expr_idx] = hdNegSlope[i][intv_idx][expr_idx] + 1 if A2 < 0 else hdNegSlope[i][intv_idx][expr_idx]

                        if distIntv[intv_idx] == 60 and r21 > bdmaxR2['R2']:
                            bdmaxR2['R2'] = r21
                            bdmaxR2['eeg'] = chan
                            bdmaxR2['start'] = start
                            bdmaxR2['fullX'] = fullX
                            bdmaxR2['fullY1'] = fullY1
                            bdmaxR2['X1'] = X1
                            bdmaxR2['Y11'] = Y11
                            bdmaxR2['A'] = A1
                            bdmaxR2['B'] = B1
                            bdmaxR2['Yline'] = Y11
                            bdmaxR2['corr'] = corr1
                            bdmaxR2['p'] = p1
                        if distIntv[intv_idx] == 60 and r22 > hdmaxR2['R2']:
                            hdmaxR2['R2'] = r22
                            hdmaxR2['eeg'] = chan
                            hdmaxR2['start'] = start
                            hdmaxR2['fullX'] = fullX
                            hdmaxR2['fullY2'] = fullY2
                            hdmaxR2['X2'] = X2
                            hdmaxR2['Y21'] = Y21
                            hdmaxR2['A'] = A2
                            hdmaxR2['B'] = B2
                            hdmaxR2['Yline'] = Y21
                            hdmaxR2['corr'] = corr2
                            hdmaxR2['p'] = p2

        del pickedEEG

        '''plot bd'''
        plt.figure(figsize=(12, 8))
        plt.scatter(bdmaxR2['fullX'], bdmaxR2['fullY1'], c='red')
        for anno_idx in range(len(seegOrder)):
            plt.annotate(seegOrder[anno_idx], xy=(bdmaxR2['fullX'][anno_idx], bdmaxR2['fullY1'][anno_idx]), xytext=(bdmaxR2['fullX'][anno_idx] + 0.1, bdmaxR2['fullY1'][anno_idx] + 0.1))
        
        plt.plot(bdmaxR2['X1'], bdmaxR2['Y11'], 'blue')
        plt.title(' '.join([patient, bdmaxR2['eeg']]))
        plt.legend()
        plt.xlabel('SEEG-EEG Electroid Physical Distance')
        plt.ylabel('Bhattacharyya Similarity')

        if saveDir is None:
            plt.show()
        else:
            plt.savefig(os.path.join(saveDir, '_'.join([patient, 'segment', chan, 'bd'])))

        plt.close()

        '''plot hd'''
        plt.figure(figsize=(12, 8))
        plt.scatter(hdmaxR2['fullX'], hdmaxR2['fullY2'], c='red')
        for anno_idx in range(len(seegOrder)):
            plt.annotate(seegOrder[anno_idx], xy=(hdmaxR2['fullX'][anno_idx], hdmaxR2['fullY2'][anno_idx]), xytext=(hdmaxR2['fullX'][anno_idx] + 0.1, hdmaxR2['fullY2'][anno_idx] + 0.1))
        
        plt.plot(hdmaxR2['X2'], hdmaxR2['Y21'])
        plt.title(' '.join([patient, hdmaxR2['eeg']]))
        plt.legend()
        plt.xlabel('SEEG-EEG Electroid Physical Distance')
        plt.ylabel('Hellinger Similarity')

        if saveDir is None:
            plt.show()
        else:
            plt.savefig(os.path.join(saveDir, '_'.join([patient, 'segment', chan, 'hd'])))

        plt.close()

    bdNegRate = bdNegSlope / n_segment
    hdNegRate = hdNegSlope / n_segment

    return {'mean bdNegRate across exper': np.mean(bdNegRate, axis=2), 'mean hdNegRate across epxer': np.mean(hdNegRate, axis=2),
            'std bdNegRate across exper': np.std(bdNegRate, axis=2),
            'std hdNegRate across epxer': np.std(hdNegRate, axis=2),
            'nSEEGperIntv': nSEEGperIntv}


def calculate_rmse(real_path, fake_path, normalizer, method='GAN', is_IF=False, aggregate=True, save_path=None, baseline=None):
    '''

    :param aggregate: whether the results of all patients are aggregated, default as True
    :param real_path: the path of real seeg data
    :param fake_path: the path of generated seeg data
    :param normalizer: normalizer
    :param z_score_path: 
    :param method: str | 'EEG','GAN', 'EUD_temporal', 'EUD_freq', 'spline', 'asae'
    :param save_path: the directory to be saved
    
    :return: result
    '''

    dataset = make_dataset(fake_path)
    if method in ['GAN', 'asae', 'ae', 'pghi', 'phase', 'EEGGAN']:
        dataset = list(filter(lambda x: 'fake' in x, dataset))

    n_band = len(bands)
    patient_result = {}

    for i, f_name in enumerate(dataset):

        print("iteration %d: " % i)

        f_n = os.path.basename(dataset[i])
        f_n = f_n.split('.')[0]
        patient = f_n.split('_')[0]
        if patient not in patient_result.keys():
            patient_result[patient] = {}
            patient_result[patient]['mag_rmse'] = [[] for _ in range(n_band)]
            patient_result[patient]['phase_rmse'] = [[] for _ in range(n_band)]
            patient_result[patient]['IF_rmse'] = [[] for _ in range(n_band)]
            patient_result[patient]['mag_mae'] = []
            patient_result[patient]['phase_mae'] = []
            patient_result[patient]['IF_mae'] = []
            patient_result[patient]['mag_cov'] = []
            patient_result[patient]['mag_row_cov'] = []
            patient_result[patient]['mag_col_cov'] = []
            patient_result[patient]['phase_cov'] = []
            patient_result[patient]['IF_cov'] = []
            patient_result[patient]['IF_row_cov'] = []
            patient_result[patient]['IF_col_cov'] = []
            patient_result[patient]['mag_mssmi'] = []
            patient_result[patient]['phase_mssmi'] = []
            patient_result[patient]['IF_mssmi'] = []
        f_n = '_'.join(f_n.split("_")[: 4])

        # read data according to method
        if method in ['GAN', 'phase', 'pghi']:
            fake = np.load(os.path.join(fake_path, f_n + '_fake_B.npy'))
        elif method in ['asae', 'ae', 'EEGGAN']:
            fake = np.load(os.path.join(fake_path, f_n + '_fake_seeg.npy'))
        elif method in ['EUD_temporal', 'EUD_freq', 'spline']:
            fake = np.load(os.path.join(fake_path, f_n.split('_')[1] + '.npy'))
        elif method in ['EEG']:
            fake = np.load(os.path.join(fake_path, f_n + '.npy'))[0]
        else:
            raise NotImplementedError

        # process inputs according to method
        if method in ['GAN', 'EUD_freq', 'ae']:
            fake = IF_to_eeg(fake, normalizer, iseeg=False, is_IF=is_IF)[0]
        elif method in ['asae', 'EEGGAN']:
            fake = normalizer.denormalize_temporal(fake, is_eeg=False)

        real = np.load(os.path.join(real_path, patient, f_n + '.npy'))[0]  
        if len(real) > fake.shape[0]:
            real = real[: fake.shape[0]]
        r_temp, r_mag, _, _ = get_time_freq_by_band(real, 0, conf.seeg_ceiling_freq, iseeg=False)  

        fake = fake.astype(np.float64)  # transfer original signal to float64 for the fake signal is generated float32

        if method in ['EEG', 'EUD_temporal', 'spline', 'asae', 'EEGGAN']:
            fake, _, _, _ = get_time_freq_by_band(fake, 0, conf.seeg_ceiling_freq, iseeg=False)

        for j, tup in enumerate(bands):

            # calculate hamming distance
            _, r_mag, r_p, r_IF = get_time_freq_by_band(real, *tup, iseeg=False)
            _, f_mag, f_p, f_IF = get_time_freq_by_band(fake, *tup, iseeg=False)

            mag_rmse = np.linalg.norm(r_mag - f_mag, ord=2) / r_mag.size ** -0.5
            patient_result[patient]['mag_rmse'][j].append(np.log(mag_rmse))
            phase_rmse = np.linalg.norm(r_p - f_p, ord=2) / r_p.size ** -0.5
            patient_result[patient]['phase_rmse'][j].append(np.log(phase_rmse))
            IF_rmse = np.linalg.norm(r_IF - f_IF, ord=2) / r_IF.size ** -0.5
            patient_result[patient]['IF_rmse'][j].append(np.log(IF_rmse))

            if j == len(bands) - 1:
                patient_result[patient]['mag_mae'].append(np.linalg.norm(r_mag - f_mag, ord=1) / r_mag.size ** -1)
                patient_result[patient]['phase_mae'].append(np.linalg.norm(r_p - f_p, ord=1) / r_p.size ** -1)
                patient_result[patient]['IF_mae'].append(np.linalg.norm(r_IF - f_IF, ord=1) / r_IF.size ** -1)
                patient_result[patient]['mag_cov'].append(pearsonr(r_mag.flatten(), f_mag.flatten())[0])
                patient_result[patient]['mag_row_cov'].append(
                    np.sum((r_mag - np.mean(r_mag, axis=1, keepdims=True)) * (f_mag - np.mean(f_mag, axis=1, keepdims=True))) / (r_mag.shape[0] * (r_mag.shape[1] - 1)))
                patient_result[patient]['mag_col_cov'].append(
                    np.sum((r_mag - np.mean(r_mag, axis=0, keepdims=True)) * (
                                f_mag - np.mean(f_mag, axis=0, keepdims=True))) / ((r_mag.shape[0] - 1) * r_mag.shape[1]))
                patient_result[patient]['phase_cov'].append(np.sum((r_p - f_p.mean()) * (r_p - f_p.mean())) / (r_p.size - 1))
                patient_result[patient]['IF_cov'].append(np.sum((r_IF - r_IF.mean()) * (f_IF - f_IF.mean())) / (r_IF.size - 1))
                patient_result[patient]['IF_row_cov'].append(
                    np.sum((r_IF - np.mean(r_IF, axis=1, keepdims=True)) * (
                                f_IF - np.mean(f_IF, axis=1, keepdims=True))) / (r_IF.shape[0] * (r_IF.shape[1] - 1)))
                patient_result[patient]['IF_col_cov'].append(
                    np.sum((r_IF - np.mean(r_IF, axis=0, keepdims=True)) * (
                            f_IF - np.mean(f_IF, axis=0, keepdims=True))) / ((r_IF.shape[0] - 1) * r_IF.shape[1]))
                patient_result[patient]['mag_mssmi'].append(measure.compare_ssim(r_mag, f_mag, gaussian_weights=True))
                patient_result[patient]['phase_mssmi'].append(measure.compare_ssim(r_p, f_p, gaussian_weights=True))
                patient_result[patient]['IF_mssmi'].append(measure.compare_ssim(r_IF, f_IF, gaussian_weights=True))

    result = None
    
    if aggregate:

        mag_rmse_array = [[] for _ in range(n_band)]
        phase_rmse_array = [[] for _ in range(n_band)]
        IF_rmse_array = [[] for _ in range(n_band)]
        mag_cov_array = []
        mag_row_cov_array = []
        mag_col_cov_array = []
        phase_cov_array = []
        IF_cov_array = []
        IF_row_cov_array = []
        IF_col_cov_array = []
        mag_mssmi_array = []
        phase_mssmi_array = []
        IF_mssmi_array = []

        for _, d in patient_result.items():

            mag_cov_array += d['mag_cov']
            mag_row_cov_array += d['mag_row_cov']
            mag_col_cov_array += d['mag_col_cov']
            phase_cov_array += d['phase_cov']
            IF_cov_array += d['IF_cov']
            IF_row_cov_array += d['IF_row_cov']
            mag_col_cov_array += d['IF_col_cov']
            mag_mssmi_array = d['mag_mssmi']
            phase_mssmi_array = d['phase_mssmi']
            IF_mssmi_array = d['IF_mssmi']

            for i in range(n_band):
                mag_rmse_array[i].extend(d['mag_rmse'][i])
                phase_rmse_array[i].extend(d['phase_rmse'][i])
                IF_rmse_array[i].extend(d['IF_rmse'][i])

        if baseline is not None:
            mag_mssmi_array = np.asarray(mag_mssmi_array)
            mag_mssmi_array = 1 - mag_mssmi_array
            mag_mssmi_array /= (1-0.9995261686407153)
            mag_mssmi_array = np.log2(mag_mssmi_array)

        mag_rmse_mean = np.mean(mag_rmse_array, axis=1)
        mag_rmse_std = np.std(mag_rmse_array, axis=1, ddof=1)
        phase_rmse_mean = np.mean(phase_rmse_array, axis=1)
        phase_rmse_std = np.std(phase_rmse_array, axis=1, ddof=1)
        IF_rmse_mean = np.mean(IF_rmse_array, axis=1)
        IF_rmse_std = np.std(IF_rmse_array, axis=1, ddof=1)
        mag_cov_mean = np.mean(mag_cov_array)
        mag_cov_std = np.std(mag_cov_array, ddof=1)
        mag_row_cov_mean = np.mean(mag_row_cov_array)
        mag_row_cov_std = np.std(mag_row_cov_array, ddof=1)
        mag_col_cov_mean = np.mean(mag_col_cov_array)
        mag_col_cov_std = np.std(mag_col_cov_array, ddof=1)
        phase_cov_mean = np.mean(phase_cov_array)
        phase_cov_std = np.std(phase_cov_array, ddof=1)
        IF_cov_mean = np.mean(IF_cov_array)
        IF_cov_std = np.std(IF_cov_array, ddof=1)
        IF_row_cov_mean = np.mean(IF_row_cov_array)
        IF_row_cov_std = np.std(IF_row_cov_array, ddof=1)
        IF_col_cov_mean = np.mean(IF_col_cov_array)
        IF_col_cov_std = np.std(IF_col_cov_array, ddof=1)
        mag_mssmi_mean = np.mean(mag_mssmi_array)
        mag_mssmi_std = np.std(mag_mssmi_array, ddof=1)
        phase_mssmi_mean = np.mean(phase_mssmi_array)
        phase_mssmi_std = np.std(phase_mssmi_array, ddof=1)
        IF_mssmi_mean = np.mean(IF_mssmi_array)
        IF_mssmi_std = np.std(IF_mssmi_array, ddof=1)

        result = {'mag_rmse_mean': mag_rmse_mean, 'mag_rmse_std': mag_rmse_std,
                  'phase_rmse_mean': phase_rmse_mean, 'phase_rmse_std': phase_rmse_std,
                  'IF_rmse_mean': IF_rmse_mean, 'IF_rmse_std': IF_rmse_std,
                  'mag_cov_mean': mag_cov_mean, 'mag_cov_std': mag_cov_std,
                  'mag_row_cov_mean': mag_row_cov_mean, 'mag_row_cov_std': mag_row_cov_std,
                  'mag_col_cov_mean': mag_col_cov_mean, 'mag_col_cov_std': mag_col_cov_std,
                  'phase_cov_mean': phase_cov_mean, 'phase_cov_std': phase_cov_std,
                  'IF_cov_mean': IF_cov_mean, 'IF_cov_std': IF_cov_std,
                  'IF_row_cov_mean': IF_row_cov_mean, 'IF_row_cov_std': IF_row_cov_std,
                  'IF_col_cov_mean': IF_col_cov_mean, 'IF_col_cov_std': IF_col_cov_std,
                  'mag_mssmi_mean': mag_mssmi_mean, 'mag_mssmi_std': mag_mssmi_std,
                  'phase_mssmi_mean': phase_mssmi_mean, 'phase_mssmi_std': phase_mssmi_std,
                  'IF_mssmi_mean': IF_mssmi_mean, 'IF_mssmi_std': IF_mssmi_std
                  }
    else:
        result = {}
        for p, d in patient_result.items():

            ori_mssmi_mean = np.mean(d['mag_mssmi'])
            if baseline is not None:
                d['mag_mssmi'] = np.asarray(d['mag_mssmi'])
                d['mag_mssmi'] = 1 - d['mag_mssmi']
                d['mag_mssmi'] /= (1-0.9995261686407153)
                mssmi_sum = d['mag_mssmi'].sum()
                mssmi_product = d['mag_mssmi'].prod()
                d['mag_mssmi'] = np.log2(d['mag_mssmi'])

            mag_rmse_mean = np.mean(d['mag_rmse'], axis=1)
            mag_rmse_std = np.std(d['mag_rmse'], axis=1, ddof=1)
            phase_rmse_mean = np.mean(d['phase_rmse'], axis=1)
            phase_rmse_std = np.std(d['phase_rmse'], axis=1, ddof=1)
            IF_rmse_mean = np.mean(d['IF_rmse'], axis=1)
            IF_rmse_std = np.std(d['IF_rmse'], axis=1, ddof=1)
            mag_cov_mean = np.mean(d['mag_cov'])
            mag_cov_std = np.std(d['mag_cov'], ddof=1)
            mag_row_cov_mean = np.mean(d['mag_row_cov'])
            mag_row_cov_std = np.std(d['mag_row_cov'], ddof=1)
            mag_col_cov_mean = np.mean(d['mag_col_cov'])
            mag_col_cov_std = np.std(d['mag_col_cov'], ddof=1)
            phase_cov_mean = np.mean(d['phase_cov'])
            phase_cov_std = np.std(d['phase_cov'], ddof=1)
            IF_cov_mean = np.mean(d['IF_cov'])
            IF_cov_std = np.std(d['IF_cov'], ddof=1)
            IF_row_cov_mean = np.mean(d['IF_row_cov'])
            IF_row_cov_std = np.std(d['IF_row_cov'], ddof=1)
            IF_col_cov_mean = np.mean(d['IF_col_cov'])
            IF_col_cov_std = np.std(d['IF_col_cov'], ddof=1)
            mag_mssmi_mean = np.mean(d['mag_mssmi'])
            mag_mssmi_std = np.std(d['mag_mssmi'], ddof=1)
            phase_mssmi_mean = np.mean(d['phase_mssmi'])
            phase_mssmi_std = np.std(d['phase_mssmi'], ddof=1)
            IF_mssmi_mean = np.mean(d['IF_mssmi'])
            IF_mssmi_std = np.std(d['IF_mssmi'], ddof=1)

            result[p] = {'mag_rmse_mean': mag_rmse_mean, 'mag_rmse_std': mag_rmse_std,
                  'phase_rmse_mean': phase_rmse_mean, 'phase_rmse_std': phase_rmse_std,
                  'IF_rmse_mean': IF_rmse_mean, 'IF_rmse_std': IF_rmse_std,
                  'mag_cov_mean': mag_cov_mean, 'mag_cov_std': mag_cov_std,
                  'mag_row_cov_mean': mag_row_cov_mean, 'mag_row_cov_std': mag_row_cov_std,
                  'mag_col_cov_mean': mag_col_cov_mean, 'mag_col_cov_std': mag_col_cov_std,
                  'phase_cov_mean': phase_cov_mean, 'phase_cov_std': phase_cov_std,
                  'IF_cov_mean': IF_cov_mean, 'IF_cov_std': IF_cov_std,
                  'IF_row_cov_mean': IF_row_cov_mean, 'IF_row_cov_std': IF_row_cov_std,
                  'IF_col_cov_mean': IF_col_cov_mean, 'IF_col_cov_std': IF_col_cov_std,
                  'mag_mssmi_mean': mag_mssmi_mean, 'mag_mssmi_std': mag_mssmi_std,
                  'phase_mssmi_mean': phase_mssmi_mean, 'phase_mssmi_std': phase_mssmi_std,
                  'IF_mssmi_mean': IF_mssmi_mean, 'IF_mssmi_std': IF_mssmi_std,
                  'ori_mssmi_mean': ori_mssmi_mean, 'mssmi_sum': mssmi_sum, 'mssmi_prod': mssmi_product
                  }

    if save_path is not None:
        np.save(save_path, result)

    return result


def compare_by_interval(real_path, fake_path, normalizer, n_interval, method='GAN', is_IF=False, aggregate=True, save_path=None):
    '''

    :param aggregate: whether the results of all patients are aggregated, default as True
    :param real_path: the path of real seeg data
    :param fake_path: the path of generated seeg data
    :param normalizer: normalizer
    :param z_score_path: 
    :param method: str | 'EEG','GAN', 'EUD_temporal', 'EUD_freq', 'spline', 'asae'
    :param save_path: the directory to be saved
    
    :return: result
    '''

    dataset = make_dataset(fake_path)
    if method in ['GAN', 'asae', 'ae', 'pghi', 'phase', 'EEGGAN']:
        dataset = list(filter(lambda x: 'fake' in x, dataset))

    patient_result = {}

    for i, f_name in enumerate(dataset):

        f_n = os.path.basename(dataset[i])
        f_n = f_n.split('.')[0]
        patient = f_n.split('_')[0]
        if patient not in patient_result.keys():
            patient_result[patient] = {}
            patient_result[patient]['dtw_interval'] = [[] for _ in range(n_interval)]
            patient_result[patient]['rmse_interval'] = [[] for _ in range(n_interval)]
        f_n = '_'.join(f_n.split("_")[: 4])

        # read data according to method
        if method in ['GAN', 'phase', 'pghi']:
            fake = np.load(os.path.join(fake_path, f_n + '_fake_B.npy'))
        elif method in ['asae', 'ae', 'EEGGAN']:
            fake = np.load(os.path.join(fake_path, f_n + '_fake_seeg.npy'))
        elif method in ['EUD_temporal', 'EUD_freq', 'spline']:
            fake = np.load(os.path.join(fake_path, f_n.split('_')[1] + '.npy'))
        elif method in ['EEG']:
            fake = np.load(os.path.join(fake_path, f_n + '.npy'))[0]
        else:
            raise NotImplementedError

        # process inputs according to method
        if method in ['GAN', 'EUD_freq', 'ae']:
            fake = IF_to_eeg(fake, normalizer, iseeg=False, is_IF=is_IF)[0]
        elif method in ['asae', 'EEGGAN']:
            fake = normalizer.denormalize_temporal(fake, is_eeg=False)

        real = np.load(os.path.join(real_path, patient, f_n + '.npy'))[0]  
        if len(real) > fake.shape[0]:
            real = real[: fake.shape[0]]
        r_temp, r_mag, _, _ = get_time_freq_by_band(real, 0, conf.seeg_ceiling_freq, iseeg=False)  

        fake = fake.astype(np.float64)  # transfer original signal to float64 for the fake signal is generated float32

        if method in ['EEG', 'EUD_temporal', 'spline', 'asae', 'EEGGAN']:
            fake, _, _, _ = get_time_freq_by_band(fake, 0, conf.seeg_ceiling_freq, iseeg=False)

        for j in range(n_interval):

            start = j * (conf.signal_len // n_interval)
            end = min((j + 1) * (conf.signal_len // n_interval), conf.signal_len)
            
            rmse_dist = np.linalg.norm(r_temp[start: end] - fake[start: end], ord=2) / (end - start) ** 0.5
            
            patient_result[patient]['rmse_interval'][j].append(rmse_dist)

    result = None
    if aggregate:
        rmse_interval_array = [[] for _ in range(n_interval)]

        for _, d in patient_result.items():
            for i in range(n_interval):
                rmse_interval_array[i].extend(d['rmse_interval'][i])

        rmse_interval_mean = np.mean(rmse_interval_array, axis=1)
        rmse_interval_std = np.std(rmse_interval_array, axis=1, ddof=1)

        result = {'rmse_interval_mean': rmse_interval_mean, 'rmse_interval_std': rmse_interval_std}
    
    else:
        result = {}
        for p, d in patient_result.items():
            rmse_interval_mean = np.mean(d['rmse_interval'], axis=1)
            rmse_interval_std = np.std(d['rmse_interval'], axis=1, ddof=1)

            result[p] = {'rmse_interval_mean': rmse_interval_mean, 'rmse_interval_std': rmse_interval_std}

    if save_path is not None:
        np.save(save_path, result)

    return result


def bestK_results(real_path, fake_path, normalizer_ls, topK, method='GAN', is_IF=False, aggregateLoc=True, saveDir=None):

    dataset = []
    for fp in fake_path:
        fp_dataset = make_dataset(fp)
        if method in ['GAN', 'asae', 'ae', 'pghi', 'phase', 'EEGGAN']:
            fp_dataset = list(filter(lambda x: 'fake' in x, fp_dataset))
        dataset.extend(fp_dataset)

    patient_result = {}
    patient_result['temporal_dist'] = {}
    patient_result['rmse_dist'] = {}
    patient_result['psd_dist'] = {}
    patient_result['hd'] = {}
    patient_result['bd'] = {}
    grouped_files = {}

    for i, f_name in enumerate(dataset):

        print("iteration %d: " % i)

        f_n = os.path.basename(dataset[i])
        f_n = f_n.split('.')[0]
        patient = f_n.split('_')[0]
        EEG_chan = f_n.split('_')[1]
        f_n = '_'.join(f_n.split("_")[: 4])
        normalizer = normalizer_ls[patient]
        for k, v in patient_result.items():
            if patient not in v.keys():
                patient_result[k][patient] = {}
                grouped_files[patient] = {}
            if EEG_chan not in patient_result[k][patient].keys():
                patient_result[k][patient][EEG_chan] = []
                grouped_files[patient][EEG_chan] = []

        # read data
        fake = np.load(f_name)

        # process inputs according to method
        if method in ['GAN', 'EUD_freq', 'ae']:
            fake = IF_to_eeg(fake, normalizer, iseeg=False, is_IF=is_IF)[0]
        elif method in ['asae', 'EEGGAN']:
            fake = normalizer.denormalize_temporal(fake, is_eeg=False)

        real = np.load(os.path.join(real_path, patient, f_n + '.npy'))[0]  
        if len(real) > fake.shape[0]:
            real = real[: fake.shape[0]]
        r_temp, r_mag, _, _ = get_time_freq_by_band(real, 0, conf.seeg_ceiling_freq, iseeg=False)  

        fake = fake.astype(np.float64)  # transfer original signal to float64 for the fake signal is generated float32

        if method in ['EEG', 'EUD_temporal', 'spline', 'asae', 'EEGGAN']:
            fake, _, _, _ = get_time_freq_by_band(fake, 0, conf.seeg_ceiling_freq, iseeg=False)

        dtw_dist = dtw.distance_fast(r_temp, fake, use_pruning=True)
        rmse_dist = np.linalg.norm(r_temp - fake, ord=2) / len(r_temp) ** 0.5
        patient_result['temporal_dist'][patient][EEG_chan].append(dtw_dist)
        patient_result['rmse_dist'][patient][EEG_chan].append(rmse_dist)

        # calculate hamming distance
        _, r_mag, r_p, r_IF = get_time_freq_by_band(real, *bands[-1], iseeg=False)
        _, f_mag, f_p, f_IF = get_time_freq_by_band(fake, *bands[-1], iseeg=False)

        # calculate psd distance
        r_psd = mne.time_frequency.psd_array_welch(real, conf.seeg_sf, fmin=bands[-1][0], fmax=bands[-1][1], n_fft=conf.seeg_n_fft)
        f_psd = mne.time_frequency.psd_array_welch(fake, conf.seeg_sf, fmin=bands[-1][0], fmax=bands[-1][1], n_fft=conf.seeg_n_fft)
        dist = np.linalg.norm(np.asarray(r_psd[0]) - np.asarray(f_psd[0]), ord=2) / len(r_psd[0]) ** 0.5
        patient_result['psd_dist'][patient][EEG_chan].append(dist)

        # calculate the hamming distance of phase/IF
        r_psd_prob = psd_probability(r_psd[0], conf.seeg_n_fft, real.shape[0])
        f_psd_prob = psd_probability(f_psd[0], conf.seeg_n_fft, real.shape[0])
        hd = Hellinger_Distance(r_psd_prob, f_psd_prob)
        bd = Bhattacharyya_Distance(r_psd_prob, f_psd_prob)
        patient_result['hd'][patient][EEG_chan].append(hd)
        patient_result['bd'][patient][EEG_chan].append(bd)
        grouped_files[patient][EEG_chan].append(f_name)

    topKresults = {}
    topKresults['temporal_dist'] = {}
    topKresults['rmse_dist'] = {}
    topKresults['psd_dist'] = {}
    topKresults['hd'] = {}
    topKresults['bd'] = {}
    sortedresults = {}
    sortedresults['temporal_dist'] = {}
    sortedresults['rmse_dist'] = {}
    sortedresults['psd_dist'] = {}
    sortedresults['hd'] = {}
    sortedresults['bd'] = {}

    for metric, p_dict in patient_result.items():

        for patient, chan_ls in p_dict.items():
            if aggregateLoc:
                aggregated_results = []
                aggregated_file_names = []
                for chan, ls in chan_ls.items():
                    aggregated_results.extend(ls)
                    aggregated_file_names.extend(grouped_files[patient][chan])

                indice = np.asarray(aggregated_results).argsort()
                sortedresults[metric][patient] = [aggregated_file_names[i] for i in indice]
                topKresults[metric][patient] = sortedresults[metric][patient][:topK]
            else:
                sortedresults[metric][patient] = {}
                topKresults[metric][patient] = {}
                for chan, ls in chan_ls.items():
                    indice = np.asarray(ls).argsort()
                    sortedresults[metric][patient][chan] = [grouped_files[patient][chan][i] for i in indice]
                    topKresults[metric][patient][chan] = sortedresults[metric][patient][chan][: topK]

    if saveDir is not None:
        topK_save_name = 'top{}Perturbation'.format(topK)
        sorted_save_name = 'sortedPerturbation'
        if aggregateLoc:
            topK_save_name += 'aggreateLoc'
            sorted_save_name += 'aggreateLoc'
        np.save(os.path.join(saveDir, topK_save_name), topKresults)
        np.save(os.path.join(saveDir, sorted_save_name), sortedresults)

    return topKresults