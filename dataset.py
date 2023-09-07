import torch.utils.data as data
import numpy as np
from utils.utils import process_feat
import torch
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from pathlib import Path
import pickle

class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False, normal_only=False):
        self.is_normal = is_normal
        self.dataset = args.dataset
        self.args = args

        if self.dataset in ['shanghai', "sh"]:
            if test_mode:
                self.rgb_list_file = f'list/{args.visual.lower()}/shanghai-{args.visual}-test-10crop.list'
                ds = "sh_test"
            else:
                self.rgb_list_file = f'list/{args.visual.lower()}/shanghai-{args.visual}-train-10crop.list'
                ds = "sh_train"
        elif self.dataset == "ucf":
            if test_mode:
                self.rgb_list_file = f'list/{args.visual.lower()}/ucf-{args.visual}-test.list'
                ds = "ucf_test"
            else:
                self.rgb_list_file = f'list/{args.visual.lower()}/ucf-{args.visual}.list'
                ds = "ucf_train"
        elif self.dataset == "xd":
            if test_mode:
                self.rgb_list_file = f'list/{args.visual.lower()}/xd-{args.visual}-test.list'
                ds = "xd_test"
            else:
                self.rgb_list_file = f'list/{args.visual.lower()}/xd-{args.visual}-train.list'
                ds = "xd_train"
        else:
            raise SystemError("Check main.py --dataset")

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None

        self.file_management = {
            "sh": 63,
            "shanghai": 63,
            "ucf": 810,
            "xd": 1905
        }

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))

        if self.test_mode is False:
            if self.dataset in ["sh", 'shanghai']:
                if self.is_normal:
                    self.list = self.list[63:]
                    assert len(self.list) == 175
                    # print(self.list)
                else:
                    self.list = self.list[:63]
                    assert len(self.list) == 63
                    # print(self.list)

            elif self.dataset == 'ucf':
                if self.is_normal:
                    self.list = self.list[810:]
                    assert len(self.list) == 800
                    # print(self.list)
                else:
                    self.list = self.list[:810]
                    assert len(self.list) == 810
                    # print(self.list)

            elif self.dataset == 'xd':
                if self.is_normal:
                    self.list = self.list[1905:]
                    assert len(self.list) == 2048
                    # print(self.list)
                else:
                    self.list = self.list[:1905]
                    assert len(self.list) == 1905
                    # print(self.list)

    def __getitem__(self, index):
        path = self.list[index].strip('\n')

        label = self.get_label()  # get video level label 0/1

        features = np.load(path, allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        # Instead of 10-crop snippet feature, let one image represent all
        # If it enters, it doesn't do tencrop
        if self.args.visual.lower() not in ["i3d", "c3d"] and len(features.shape) != 3:
            features = features.reshape((*features.shape[:-1],1,features.shape[-1]))

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features
        else:
            # currently: (31, 10, 2048)
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, B, T, F] -> (10, 31, 2048)
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)

            return divided_features, label                

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
        