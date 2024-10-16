import mne
import os
import numpy as np
from .numpy_tools import load_from_ndarray, save_as_ndarray, make_dataset, draw_comparable_figure
from scipy.stats import wasserstein_distance
import random
import shutil
import librosa
import matplotlib.pyplot as plt
import scipy
import scipy.io as io
import ntpath
from utils import phase_operation

# channels
seeg_ch = ['EEG A1-Ref-0', 'EEG A2-Ref-0', 'POL A3', 'POL A4', 'POL A5', 'POL A6', 'POL A7', 'POL A8', 'POL A9',
           'POL A10', 'POL A11', 'POL A12', 'POL A13', 'POL A14', 'POL A15', 'POL A16', 'POL B1', 'POL B2', 'POL B3',
           'POL B4', 'POL B5', 'POL B6', 'POL B7', 'POL B8', 'POL B9', 'POL B10', 'POL B11', 'POL B12', 'POL B13',
           'POL B14', 'EEG C1-Ref', 'EEG C2-Ref', 'EEG C3-Ref-0', 'EEG C4-Ref-0', 'EEG C5-Ref', 'EEG C6-Ref', 'POL C7',
           'POL C8', 'POL C9', 'POL C10', 'POL C11', 'POL C12', 'POL C13', 'POL C14', 'POL D1', 'POL D2', 'POL D3',
           'POL D4', 'POL D5', 'POL D6', 'POL D7', 'POL D8', 'POL D9', 'POL D10', 'POL D11', 'POL D12', 'POL D13',
           'POL D14', 'POL D15', 'POL D16', 'POL E1', 'POL E2', 'POL E3', 'POL E4', 'POL E5', 'POL E6', 'POL E7',
           'POL E8', 'POL E9', 'POL E10', 'POL E11', 'POL E12', 'POL E13', 'POL E14', 'EEG F1-Ref', 'EEG F2-Ref',
           'EEG F3-Ref-0', 'EEG F4-Ref-0', 'EEG F5-Ref', 'EEG F6-Ref', 'EEG F7-Ref-0', 'EEG F8-Ref-0', 'EEG F9-Ref',
           'EEG F10-Ref', 'POL F11', 'POL F12', 'POL G1', 'POL G2', 'POL G3', 'POL G4', 'POL G5', 'POL G6', 'POL G7',
           'POL G8', 'POL G9', 'POL G10', 'POL G11', 'POL G12', 'POL H1', 'POL H2', 'POL H3', 'POL H4', 'POL H5',
           'POL H6', 'POL H7', 'POL H8', 'POL H9', 'POL H10', 'POL H11', 'POL H12', 'POL H13', 'POL H14', 'POL H15',
           'POL H16', 'POL K1', 'POL K2', 'POL K3', 'POL K4', 'POL K5', 'POL K6', 'POL K7', 'POL K8', 'POL K9',
           'POL K10', 'POL K11', 'POL K12', 'POL K13', 'POL K14', 'POL K15', 'POL K16']
eeg_ch = ['EEG Fp1-Ref', 'EEG F7-Ref-1', 'EEG T3-Ref', 'EEG T5-Ref', 'EEG O1-Ref', 'EEG F3-Ref-1', 'EEG C3-Ref-1',
          'EEG P3-Ref', 'EEG FZ-Ref', 'EEG CZ-Ref', 'EEG PZ-Ref', 'EEG OZ-Ref', 'EEG Fp2-Ref', 'EEG F8-Ref-1',
          'EEG T4-Ref', 'EEG T6-Ref', 'EEG O2-Ref', 'EEG F4-Ref-1', 'EEG C4-Ref-1', 'EEG P4-Ref', 'EEG A1-Ref-1',
          'EEG A2-Ref-1']

exclusions = ['EEG OZ-Ref', 'EEG T3-Ref', 'EEG T6-Ref', 'EEG A1-Ref-1', 'EEG A2-Ref-1']


class Configuration:

    def __init__(self, transform='specgrams'):

        self.signal_len = 1024  # 1536
        self.eeg_sf = 64
        self.eeg_n_fft = 256  # 512
        self.eeg_win_len = 256
        self.eeg_hop = 8  # 12  # 8
        self.eeg_ceiling_freq = 32  # 28
        self.seeg_sf = 64
        self.seeg_n_fft = 256  # 512
        self.seeg_win_len = 256
        self.seeg_hop = 8  # 12  # 8
        self.seeg_ceiling_freq = 32  # 28
        self.epsilon = 1.0e-7
        self.eeg_pos = {'Fp1': (-1, 2), 'Fp2': (1, 2), 'F7': (-2, 1), 'F3': (-1, 1), 'FZ': (0, 1), 'F4': (1, 1),
                        'F8': (2, 1), 'T3': (-2, 0), 'C3': (-1, 0), 'CZ': (0, 0), 'C4': (1, 0), 'T4': (2, 0),
                        'T5': (-2, -1), 'P3': (-1, -1), 'PZ': (0, -1), 'P4': (1, -1), 'T6': (2, -1), 'O1': (-1, -2),
                        'O2': (1, -2)}
        self.seeg_mapping = {'lk': {'F1': 'FZ', 'E10': 'Fp2', 'F11': 'F4', 'B14': 'F8', 'D15': 'C4', 'H14': 'T4'},
                             'tll': {'B15': 'F7', 'K14': 'T3', 'M15': 'T4', 'E2': 'PZ'},
                             'zxl': {'G13': 'F8', 'D1': 'CZ', 'I10': 'C4', 'H14': 'T4', 'C1': 'PZ', 'F16': 'T6',
                                     'L5': 'O2'},
                             'yjh': {'E15': 'F7', 'K14': 'F8', 'L9': 'C4', 'D15': 'T3', 'J14': 'T4'}}  # establish a mapping of SEEG-to-EEG channels by patient
        self.h = 128  # 224
        self.w = 128  # 224
        if transform == 'specgrams':
            self.audio_length = (self.h - 1) * self.seeg_hop
        elif transform == 'pghi':
            self.audio_length = self.h * self.seeg_hop


conf = Configuration()


def filter_signal(raw_data, low, high, rm_pln=False):
    """
    :param raw_data: instance of mne.Raw
    :param low: float | None  the lower pass-band edge. If None the data are only low-passed.
    :param high: float | None  the upper pass-band edge. If None the data are only high-passed.
    :param rm_pln: bool  remove power line noise if True
    :return: instance of mne.Raw
    """

    if rm_pln:
        raw_data.notch_filter(np.arange(50., int(high / 50) * 50 + 1., 50), fir_design='firwin')
    raw_data.filter(low, high, fir_design='firwin')
    return raw_data


def down_sampling(raw_data, sfreq):
    """
    :param raw_data: instance of mne.Raw
    :param sfreq: New sample rate to use.
    :return: instance of mne.Raw
    """

    raw_data.resample(sfreq, npad="auto")
    return raw_data


def read_raw_signal(filename):
    if filename.endswith(".edf"):
        return mne.io.read_raw_edf(filename, preload=True)
    elif filename.endswith(".fif"):
        return mne.io.read_raw_fif(filename, preload=True)


def get_channels(raw_data):
    return raw_data.ch_names


def pick_data(raw_data, isA):
    chans = get_channels(raw_data)
    if isA:
        half = chans[:int(len(chans) / 2)]
    else:
        half = chans[int(len(chans) / 2) + 1:]
    picked_data = raw_data.pick_channels(half)
    print("Channels picked!")

    return picked_data[:, :][0]


def change_dir(path, isA, istrain=True):
    data_path = path
    if isA:
        data_path = os.path.join(data_path, 'A')
    else:
        data_path = os.path.join(data_path, 'B')
    if istrain:
        data_path = os.path.join(data_path, 'train')
    else:
        data_path = os.path.join(data_path, 'test')
    os.chdir(data_path)
    print("Current working directory is %s" % os.getcwd())


def get_next_number():
    numbers = sorted([int(os.path.splitext(os.path.basename(file))[0]) for file in make_dataset(os.getcwd())])
    if len(numbers) == 0:
        last_file_number = -1
    else:
        last_file_number = numbers[-1]

    return last_file_number + 1


def slice_data(raw_data, save_path, width, hop=None, start=0, end=None, start_number=None, prefix=''):
    """
    :param raw_data: mne.Raw format, raw eeg data or seeg data
    :param save_path: the path to be saved
    :param width: the length of segment to be sliced
    :param hop: the length between two segments
    :param start: start point, default as 0
    :param end: end point, default as terminal
    
    :return: next: the idx of next item
    """

    origin_wd = os.getcwd()
    os.chdir(save_path)

    if end is None:
        end = raw_data.get_data().shape[1]

    if hop is None:
        hop = width

    if start_number is not None:
        next = start_number
    else:
        next = get_next_number()
    raw_data = raw_data.get_data()

    for i in range(start, end, hop):
        if i + width > end:
            break

        segment = raw_data[:, i: i + width]
        save_as_ndarray(prefix + str(next), segment)
        if next % 1000 == 0:
            print("Slice %d done!" % next)
        next += 1

    print("Total pieces: %d" % next)
    os.chdir(origin_wd)

    return next


def slice_random_data(raw_data, save_path, random_starts, width, prefix=''):
    total = 0
    raw_data = raw_data.get_data()

    for i, start in enumerate(random_starts):
        segment = raw_data[:, start: start + width]
        save_as_ndarray(os.path.join(save_path, prefix + str(start)), segment)
        if total % 1000 == 0:
            print("Slice %d done!" % total)
        total += 1

    print("Total pieces: %d" % total)

    return total


def copy_random_files(n_random, seeg_src, seeg_dest, eeg_src, eeg_dest):
    dataset = make_dataset(seeg_src)
    rd = random.sample(range(0, len(dataset)), n_random)
    
    for i, f in enumerate(dataset):
        if i in rd:
            f_n = os.path.basename(f)
            shutil.move(os.path.join(eeg_src, f_n), os.path.join(eeg_dest, f_n))
            shutil.move(os.path.join(seeg_src, f_n), os.path.join(seeg_dest, f_n))
        

def get_signals_by_variance(sig_dir, top_n, decending=True, style='average'):
    """
    :param sig_dir:
    :param top_n:
    :param decending:
    :param style:  option "average" or "single", select top variance according single or average channels
    
    :return: list of ndarray
    """
    
    file_paths = make_dataset(sig_dir)
    v = []
    signals = []
    for path in file_paths:
        signal = load_from_ndarray(path)
        signals.append(signal)
        if style == 'average':
            v.append(signal.var(axis=1).mean())
        elif style == 'single':
            v.append(signal.var(axis=1).max())

    sorted_signals = [pair[1] for pair in sorted(zip(v, signals), reverse=decending)]
    return sorted_signals[: top_n]


def get_time_freq_by_band(signal, low, high, iseeg, is_IF=False):
    '''
    filter signals
    :param signal: ndarray, shape (n_times)
    :param low: low frequency
    :param high: high frequency
    
    :return: filtered data
    '''

    if iseeg:
        sf = conf.eeg_sf
        n_fft = conf.eeg_n_fft
        hop = conf.eeg_hop
    else:
        sf = conf.seeg_sf
        n_fft = conf.seeg_n_fft
        hop = conf.seeg_hop

    if high < conf.seeg_sf / 2:
        filtered = mne.filter.filter_data(signal, sf, low, high)  # time domain
    else:
        filtered = signal
    
    spec = librosa.stft(filtered, n_fft=n_fft, hop_length=hop)[
           int(low * n_fft / sf): int(high * n_fft / sf)]  # frequency domain
    magnitude = np.abs(spec)  # + conf.epsilon
    phase = np.angle(spec)
    
    IF = phase_operation.instantaneous_frequency(phase, time_axis=1)

    return filtered, magnitude, phase, IF


def time_frequency_transform(raw_data, freqs, sf, output):
    """
    :param raw_data: array
    :param freqs: array_like of float, shape (n_freqs,)  list of output frequencies
    :param output: str in ['complex', 'phase', 'power', 'avg_power', 'avg_power_itc' ]
    
    :return: Time frequency transform of epoch_data.  If output is in  ['complex', 'phase', 'power'], \
    then shape of out is (n_epochs, n_chans, n_freqs, n_times), else it is (n_chans, n_freqs, n_times).
    """

    if 'avg' not in output:
        raw_data = raw_data[np.newaxis, :]
    power = mne.time_frequency.tfr_array_morlet(raw_data, sfreq=sf,
                                                freqs=freqs, n_cycles=freqs / 2.,
                                                output=output)
    return power


def analyze_power_each_freq(power, freqs, method):
    n_chan = len(power[0][0])
    n_freq = len(freqs)
    result = [0 for _ in range(n_freq)]
    for chan in power[0]:
        for i in range(n_freq):
            if method == 'mean':
                result[i] += chan[i].mean()
            elif method == 'max':
                result[i] = max(result[i], chan[i].max())
    if method == 'mean':
        result = [1.0 * i / n_chan for i in result]

    return result


def polar2rect(mag, phase_angle):
    """Convert polar-form complex number to its rectangular form."""
    
    temp_mag = np.zeros(mag.shape, dtype=np.complex_)
    temp_phase = np.zeros(mag.shape, dtype=np.complex_)

    for i, time in enumerate(mag):
        for j, time_id in enumerate(time):
            temp_mag[i, j] = np.complex(mag[i, j])

    for i, time in enumerate(phase_angle):
        for j, time_id in enumerate(time):
            temp_phase[i, j] = np.complex(np.cos(phase_angle[i, j]), np.sin(phase_angle[i, j]))

    return temp_mag * temp_phase


def mag_plus_phase(mag, IF, iseeg, is_IF=False):
    hop = conf.eeg_hop if iseeg else conf.seeg_hop
    n_fft = conf.eeg_n_fft if iseeg else conf.seeg_n_fft

    h, w = mag.shape
    mag = np.exp(mag) - conf.epsilon  

    if h < n_fft // 2 + 1:
        mag = np.vstack((mag, np.zeros((n_fft // 2 - h + 1, w))))
        IF = np.vstack((IF, np.zeros((n_fft // 2 - h + 1, w))))
    reconstruct_magnitude = np.abs(mag)

    if is_IF:
        reconstruct_phase_angle = np.cumsum(IF * np.pi, axis=1)  # accumulate along time dimension
    else:
        reconstruct_phase_angle = IF

    stft = polar2rect(reconstruct_magnitude, reconstruct_phase_angle)
    inverse = librosa.istft(stft, hop_length=hop, window='hann')

    return inverse


def IF_to_eeg(IF_output, normalizer, iseeg=True, is_IF=False):
    '''
    :param IF_output: output of model, format is IF
    :param normalizer: 
    
    :return: an eeg or seeg segment
    '''

    n_fft = conf.eeg_n_fft if iseeg else conf.seeg_n_fft
    h, w = IF_output.shape[-2:]
    
    if iseeg == False:
        IF_output = IF_output[np.newaxis, :]
    
    out = []
    for i in range(IF_output.shape[0]):
        if h < n_fft // 2:
            magnitude, IF = normalizer.denormalize(IF_output[i][0], IF_output[i][1], iseeg)
        
        else:
            magnitude = np.vstack((IF_output[i][0], IF_output[i][0][-1]))
            IF = np.vstack((IF_output[i][1], IF_output[i][1][-1]))  
            magnitude, IF = normalizer.denormalize(magnitude, IF, iseeg)
        out.append(mag_plus_phase(magnitude, IF, iseeg, is_IF))

    recovered = np.asarray(out)

    return recovered


def save_comparison_plots(real_dir, fake_dir, save_dir, normalizer):
    dataset = make_dataset(real_dir)
    for f_name in dataset:
        f_n = os.path.basename(f_name).split('.')[0]
        real_B = np.load(os.path.join(real_dir, f_n + '.npy'))
        fake_B = np.load(os.path.join(fake_dir, f_n + '_fake_B.npy'))
        fake_B = IF_to_eeg(fake_B, normalizer)

        draw_comparable_figure(Real=real_B, Fake=fake_B[None, :], ch_intv=0, show=False)
        plt.savefig(os.path.join(save_dir, f_n))
        plt.close()


def save_origin_mat(visuals, image_dir, image_path, normlizer):
    """Save the test data in MAT format, only use in testing phase.
    
    :param visuals:
    :param image_dir:
    :param image_path:
    :param save_size:
    
    :return:
    """

    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    for label, im_data in visuals.items():
        if 'fake' in label:
            mat_name = '%s_%s' % (name, label)  
            mat_data = im_data[0]
            mat_data = mat_data.cpu().numpy()
            mat_data = IF_to_eeg(mat_data, normlizer)
            mat_data = np.expand_dims(mat_data, axis=0)
            save_path = os.path.join(image_dir, mat_name)
            io.savemat(save_path, {'EEGvector': mat_data})


def pghi_invert(output, normalizer, postprocessor, iseeg=True):
    output = output.squeeze()

    magnitude = normalizer.denormalize(output, is_eeg=iseeg)
    recovered = postprocessor(np.vstack((magnitude, magnitude[-1])))[: conf.signal_len]

    return recovered[np.newaxis, :]
