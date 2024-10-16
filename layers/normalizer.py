import numpy as np
import torch


class DataNormalizer(object):
    def __init__(self, dataloader, dict_name, do=False, domain='freq', use_phase=True):
        self.dataloader = dataloader
        self.dic_name = dict_name
        self.domain = domain
        self.use_phase = use_phase
        # self.s_a = 0 
        # self.s_b = 0
        # self.p_a = 0
        # self.p_b = 0

        if do:
            if self.domain == 'freq':
                self._range_normalizer(s_magnitude_margin=0.8, s_IF_margin=1.0, e_magnitude_margin=0.8, e_IF_margin=1.0,
                                       e_pos_margin=1.0)  # why 0.8
            else:
                self._range_normalizer_time_domain(seeg_margin=0.8, eeg_margin=0.8)
        else:
            tmp = np.load(dict_name, allow_pickle=True).item()
            if self.domain == 'freq':
                self.s_s_a = tmp['s_s_a']
                self.s_s_b = tmp['s_s_b']
                self.e_s_a = tmp['e_s_a']
                self.e_s_b = tmp['e_s_b']
                if self.use_phase:
                    self.s_p_a = tmp['s_p_a']
                    self.s_p_b = tmp['s_p_b']
                    self.e_p_a = tmp['e_p_a']
                    self.e_p_b = tmp['e_p_b']
            else:
                self.seeg_a = tmp['seeg_a']
                self.seeg_b = tmp['seeg_b']
                self.eeg_a = tmp['eeg_a']
                self.eeg_b = tmp['eeg_b']
            # self.e_pos_a = tmp['e_pos_a']
            # self.e_pos_b = tmp['e_pos_b']
        if dataloader is not None:
            if self.domain == 'freq':
                print("s_s_a:", self.s_s_a)
                print("s_s_b:", self.s_s_b)
                print("e_s_a:", self.e_s_a)
                print("e_s_b:", self.e_s_b)
                if self.use_phase:
                    print("s_p_a:", self.s_p_a)
                    print("s_p_b:", self.s_p_b)
                    print("e_p_a:", self.e_p_a)
                    print("e_p_b:", self.e_p_b)
            else:
                print("seeg_a:", self.seeg_a)
                print("seeg_b:", self.seeg_b)
                print("eeg_a:", self.eeg_a)
                print("eeg_b:", self.eeg_b)
            # print('e_pos_a:', self.e_pos_a)
            # print('e_pos_b:', self.e_pos_b)


    def _range_normalizer(self, s_magnitude_margin, s_IF_margin, e_magnitude_margin, e_IF_margin, e_pos_margin):
        #     x = x.flatten()
        s_min_spec = float('inf')
        s_max_spec = float('-inf')
        s_min_IF = float('inf')
        s_max_IF = float('-inf')
        e_min_spec = float('inf')
        e_max_spec = float('-inf')
        e_min_IF = float('inf')
        e_max_IF = float('-inf')
        e_min_pos = float('inf')
        e_max_pos = float('-inf')

        # for batch_idx, (spec, IF, pitch_label, mel_spec, mel_IF) in enumerate(self.dataloader.train_loader):

        for batch_idx, data in enumerate(self.dataloader):

            # training mel
            seeg = data['A']
            eeg = data['B']

            s_spec = seeg[:, :1, ...]
            e_spec = eeg[:, : 1, ...]

            if s_spec.min().item() < s_min_spec: s_min_spec = s_spec.min().item()
            if s_spec.max().item() > s_max_spec: s_max_spec = s_spec.max().item()

            if e_spec.min().item() < e_min_spec: e_min_spec = e_spec.min().item()
            if e_spec.max().item() > e_max_spec: e_max_spec = e_spec.max().item()

            if self.use_phase:
                s_IF = seeg[:, 1:, ...]
                e_IF = eeg[:, 1: 2, ...]

                if s_IF.min().item() < s_min_IF: s_min_IF = s_IF.min().item()
                if s_IF.max().item() > s_max_IF: s_max_IF = s_IF.max().item()

                if e_IF.min().item() < e_min_IF: e_min_IF = e_IF.min().item()
                if e_IF.max().item() > e_max_IF: e_max_IF = e_IF.max().item()
            # e_pos = eeg[:, :, 2:, ...]

            # print("spec",spec.shape)
            # print("IF",IF.shape)

            # if e_pos.min().item() < e_min_pos: e_min_pos = e_pos.min().item()
            # if e_pos.max().item() > e_max_pos: e_max_pos = e_pos.max().item()

            # print(min_spec)
            # print(max_spec)
            # print(min_IF)
            # print(max_IF)

        # 把数据伸缩到(-magnitude_margin,magnitude_margin)间
        self.s_s_a = s_magnitude_margin * (2.0 / (s_max_spec - s_min_spec))
        self.s_s_b = s_magnitude_margin * (-2.0 * s_min_spec / (s_max_spec - s_min_spec) - 1.0)

        self.e_s_a = e_magnitude_margin * (2.0 / (e_max_spec - e_min_spec))
        self.e_s_b = e_magnitude_margin * (-2.0 * e_min_spec / (e_max_spec - e_min_spec) - 1.0)

        if self.use_phase:
            self.s_p_a = s_IF_margin * (2.0 / (s_max_IF - s_min_IF))
            self.s_p_b = s_IF_margin * (-2.0 * s_min_IF / (s_max_IF - s_min_IF) - 1.0)

            self.e_p_a = e_IF_margin * (2.0 / (e_max_IF - e_min_IF))
            self.e_p_b = e_IF_margin * (-2.0 * e_min_IF / (e_max_IF - e_min_IF) - 1.0)
        # if e_max_pos == e_min_pos:
        #     self.e_pos_a = 0
        #     self.e_pos_b = 0
        # else:
        #     self.e_pos_a = e_pos_margin * (2.0 / (e_max_pos - e_min_pos))
        #     self.e_pos_b = e_pos_margin * (-2.0 * e_min_pos / (e_max_pos - e_min_pos) - 1.0)
        # self.e_pos_a = 1
        # self.e_pos_b = 0

        tmp = {'s_s_a': self.s_s_a, 's_s_b': self.s_s_b, 'e_s_a': self.e_s_a, 'e_s_b': self.e_s_b}
        if self.use_phase:
            tmp.update({'s_p_a': self.s_p_a, 's_p_b': self.s_p_b, 'e_p_a': self.e_p_a, 'e_p_b': self.e_p_b})
            # 'e_pos_a': self.e_pos_a, 'e_pos_b': self.e_pos_b}
        np.save(self.dic_name, tmp)

    
    def _range_normalizer_time_domain(self, seeg_margin=0.8, eeg_margin=0.8):

        s_min = float('inf')
        s_max = float('-inf')
        e_min = float('inf')
        e_max = float('-inf')

        for batch_idx, data in enumerate(self.dataloader):

            # training mel
            seeg = data['A']
            eeg = data['B']

            if seeg.min().item() < s_min: s_min = seeg.min().item()
            if seeg.max().item() > s_max: s_max = seeg.max().item()

            if eeg.min().item() < e_min: e_min = eeg.min().item()
            if eeg.max().item() > e_max: e_max = eeg.max().item()

        # scale data to (-magnitude_margin, magnitude_margin)
        self.seeg_a = seeg_margin * (2.0 / (s_max - s_min))
        self.seeg_b = seeg_margin * (-2.0 * s_min / (s_max - s_min) - 1.0)
        self.eeg_a = eeg_margin * (2.0 / (e_max - e_min))
        self.eeg_b = eeg_margin * (-2.0 * e_min / (e_max - e_min) - 1.0)

        tmp = {'seeg_a': self.seeg_a, 'seeg_b': self.seeg_b, 'eeg_a': self.eeg_a, 'eeg_b': self.eeg_b}
        # 'e_pos_a': self.e_pos_a, 'e_pos_b': self.e_pos_b}
        np.save(self.dic_name, tmp)

    
    def normalize(self, feature_map, type):
        if type == 'seeg':
            if self.domain == 'freq':
                if self.use_phase:
                    a = np.asarray([self.s_s_a, self.s_p_a])[None, :, None, None]
                    b = np.asarray([self.s_s_b, self.s_p_b])[None, :, None, None]
                else:
                    a = np.asarray([self.s_s_a])[None, :, None, None]
                    b = np.asarray([self.s_s_b])[None, :, None, None]
            else:
                a = np.asarray([self.seeg_a])[None, :]
                b = np.asarray([self.seeg_b])[None, :]
        elif type == 'eeg':
            if self.domain == 'freq':
                if self.use_phase:
                    a = np.asarray([self.e_s_a, self.e_p_a])[None, :, None, None]
                    b = np.asarray([self.e_s_b, self.e_p_b])[None, :, None, None]
                else:
                    a = np.asarray([self.e_s_a])[None, :, None, None]
                    b = np.asarray([self.e_s_b])[None, :, None, None]
            else:
                a = np.asarray([self.eeg_a])[None, :]
                b = np.asarray([self.eeg_b])[None, :]

        a = torch.FloatTensor(a)  # .cuda()
        b = torch.FloatTensor(b)  # .cuda()

        feature_map = feature_map * a + b
        
        return feature_map

    
    def denormalize(self, spec, IF=None, is_eeg=True):
        if is_eeg:
            spec = (spec - self.e_s_b) / self.e_s_a
            if self.use_phase:
                IF = (IF - self.e_p_b) / self.e_p_a
        else:
            spec = (spec - self.s_s_b) / self.s_s_a
            if self.use_phase:
                IF = (IF - self.s_p_b) / self.s_p_a

        if self.use_phase:
            return spec, IF
        else:
            return spec

    
    def denormalize_temporal(self, input, is_eeg=True):
        if is_eeg:
            output = (input - self.eeg_b) / self.eeg_a
        else:
            output = (input - self.seeg_b) / self.seeg_a
        return output
