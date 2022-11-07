import logging
import torch.utils.data import DataLoader, Dataset
from typing import List, Union, Dict, Any, Mapping, Optional
import torch

logger = logging.getLogger(__name__)


calss SampleDataset(Dataset):

    def __init__(self, sample_datas, sample_labels=None):
        super().__init__()
        self.sample_datas = sample_datas
        self.sample_labels = sample_labels

    def __len__(self):
        return len(self.sample_datas)

    def __getitem__(self, idx):
        sample_data = self.sample_datas[idx]
        sample_label = self.sample_labels[idx] if self.sample_labels is not None else None
        return sample_data, sample_label


class DataProcessor:

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cache = dict()

    def process(self, flag):
        assert flag in ('train', 'test', 'dev'), 'flag need in ("train", "test", "dev")'
        if flag in self.cache: return self.cache[flag]
        datafile = self.config.data_path + '.' + flag

        logger.info('start _load_data in {}'.format(datafile))
        sample_datas, sample_labels = self._load_data(datafile)
        sample_dataset = SampleDataset(sample_datas, sample_labels)
        sample_dataloader = DataLoader(sample_dataset, batch_size=self.config.batch_size, shuffle=self.config.shuffle if flag != 'test' else False)

        self.cache[flag] = sample_dataloader
        
        return sample_dataloader

    def _load_data(self, datafile):
        sample_datas, sample_labels = [], []
        logger.info('start load file in {}'.format(datafile))
        with open(datafile, 'r') as f:
            feature_name = f.readline().strip().split(',')[2:]
            for i, line in enumerate(f):
                datas = [torch.tensor(x, dtype=torch.long) for x in line.strip().split(',')]
                sample_datas.append(dict(zip(feature_name, datas[2:])))
                sample_labels.append(datas[:2])
        logger.info('file {} line num {}; data: {} label: {}'.format(datafile, i, sample_datas[0], sample_labels[0]))
        return sample_datas, sample_labels
