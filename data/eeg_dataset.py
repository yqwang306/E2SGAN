import os.path

from data.base_dataset import BaseDataset, get_transform
from data.EEGcls.eegsegment import EEGSegment
from utils.numpy_tools import make_dataset

import numpy as np
import torch



class EEGDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, patient_ls):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        phase_dir = 'train' if opt.phase in ['train', 'trainAsTest', 'perturbation'] else opt.phase
        dataIndex = np.load(os.path.join(opt.dataroot, 'dataIndex.npy'), allow_pickle=True).item()  # depends on your data
        self.A_paths = []
        self.B_paths = []
        self.weights = []
        self.t2f = True if opt.domain == 'freq' else False

        for name in patient_ls:
            dataIdx = dataIndex[name]
            # read by patient name
            A_paths = make_dataset(os.path.join(opt.dataroot, 'A', phase_dir, name))
            B_paths = make_dataset(os.path.join(opt.dataroot, 'B', phase_dir, name))
            selected_APaths = []
            selected_BPaths = []
            
            if opt.phase == 'perturbation':  # perturbation experiment
                topKresults = np.load('/public/home/xlwang/hmq/Infos/perturbation/top100FileNamesPSD.npy',
                                      allow_pickle=True).item()  # data path
                for file in topKresults[name]:
                    f_n = os.path.basename(file)
                    f_n = f_n.split('.')[0]
                    f_n = '_'.join(f_n.split('_')[:4]) + '.npy'
                    selected_APaths.append(os.path.join(opt.dataroot, 'A', 'train', name, f_n))
                    selected_BPaths.append(os.path.join(opt.dataroot, 'B', 'train', name, f_n))

            elif phase_dir == 'train':
                for idx in dataIdx:
                    selected_APaths.append(A_paths[idx])
                    selected_BPaths.append(B_paths[idx])
            
            else:
                selected_APaths = A_paths
                selected_BPaths = B_paths

            self.A_paths += sorted(selected_APaths)
            self.B_paths += sorted(selected_BPaths)
            self.weights.append(len(selected_APaths))

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain(an EEG segment)
            B (tensor) - - its corresponding image in the target domain(an SEEG segment)
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        packed = {}
        
        # read a image given a random integer index
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        A = EEGSegment(A_path)
        B = EEGSegment(B_path)

        A_transform = get_transform(self.opt, iseeg=False, t2f=self.t2f)
        B_transform = get_transform(self.opt, iseeg=True, t2f=self.t2f)

        A = A_transform(A)  # return tensor
        B = B_transform(B)

        packed.update({'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path})

        return packed

    def __len__(self):
        """Return the total number of images in the dataset."""
        return sum(self.weights)

    def get_class_weights(self):
        return self.weights


class EEGDatasetDataloader:

    def __init__(self, opt, patient_ls):

        self.opt = opt
        self.dataset = EEGDataset(opt, patient_ls)  # create a dataset given opt.dataset_mode and other options
        
        if opt.max_dataset_size == float('inf'):
            self.dataset_size = len(self.dataset)
        else:
            self.dataset_size = opt.max_dataset_size
        
        if opt.isTrain and 'eeggan' not in opt.model and len(opt.gpu_ids) > 0:
            self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=opt.batch_size)
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return self.dataset_size

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.dataset_size:
                break
            yield data
