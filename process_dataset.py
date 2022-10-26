import numpy as np
import joblib
import re
import random
import logging
from collections import namedtuple

random.seed(2020)
np.random.seed(2020)
logging.getLogger(__name__)
data_path = 'data/sample_skeleton_{}.csv'
common_feat_path = 'data/common_features_{}.csv'
write_path = 'data/ctr_cvr'
use_columns = [
    '101',
    '121',
    '122',
    '124',
    '125',
    '126',
    '127',
    '128',
    '129',
    '205',
    '206',
    '207',
    '216',
    '508',
    '509',
    '702',
    '853',
    '301']

class process_dataset(object):
    def __init__(self):
        self.feat_map = None

    def process(self, flag='train'):
        assert flag in ('train', 'test'), 'process is "train" or "test"'

        common_feat_dict = self._get_common_dict(flag)
        sample_list, feat_map = self._get_sample_list(common_feat_dict, flag)

        if flag == 'train':
            with open(write_path + '.train', 'w') as fw1:
            with open(write_path + '.dev', 'w') as fw2:
                fw1.write('click,purchase,' + ','.join(use_columns) + '\n')
                fw2.write('click,purchase,' + ','.join(use_columns) + '\n')
                for sample in sample_list:
                    new_sample = sample[:2]
                    for value, feat in zip(sample[2:], use_columns):
                        new_sample.append(str(feat_map[feat].get(value, '0')))
                    if random.random() >= 0.9:
                        fw2.write(','.join(new_sample) + '\n')
                    else:
                        fw1.write(','.join(new_sample) + '\n')
        else:
            with open(write_path + '.test', 'w') as fw:
                fw.write('click,purchase,' + ','.join(use_columns) + '\n')
                for sample in sample_list:
                    new_sample = sample[:2]
                    for value, feat in zip(sample[2:], use_columns):
                        new_sample.append(str(feat_map[feat].get(value, '0')))
                    fw.write(','.join(new_sample) + '\n')


    def _get_sample_list(self, common_feat_dict, flag):
        sample_list = []
        sample = namedtuple('sample', ['click', 'purchase'] + ['feat_' + x for x in use_columns])
        vocabulary = dict(zip(use_columns, [{}  for _ in range(len(use_columns))]))
        with open(data_path.format(flag), 'r') as fr:
            for line in fr:
                line_list = line.strip().split(',')
                if line_list[1] == '0' and line_list[2] == '1':
                    continue
                kv = np.array(re.split('\x01|\x02|\x03', line_list[5]))
                key = kv[range(0, len(kv), 3)]
                value = kv[range(1, len(kv), 3)]
                feat_dict = dict(zip(key, value))
                feat_dict.update(common_feat_dict[line_list[3]])
                feats = line_list[1:3]
                for k in use_columns:
                    feats.append(feat_dict.get(k, '0'))
                sample_list.append(sample(*feats))
                for k, v in feat_dict.items():
                    if k in use_columns:
                        if v in vocabulary[k]:
                            vocabulary[k][v] += 1
                        else:
                            vocabulary[k][v] = 0
        new_vocabulary = dict(zip(use_columns, [set() for _ in range(len(use_columns))]))
        for k, v in vocabulary.items():
            for k1, v1 in v.items():
                if v1 > 10:
                    new_vocabulary[k].add(k1)
        vocabulary = new_vocabulary
        if flag == 'train':
            for feat in use_columns:
                feat_map[feat] = dict(
                    zip(vocabulary[feat], range(1, len(vocabulary[feat]) + 1)))
            self.feat_map = feat_map
        return sample_list, self.feat_map

    def _get_common_dict(self, flag):
        c = 0
        common_feat_dict = {}
        with open(common_feat_path.format(flag), 'r') as fr:
            for line in fr:
                line_list = line.strip().split(',')
                kv = np.array(re.split('\x01|\x02|\x03', line_list[2]))
                key = kv[range(0, len(kv), 3)]
                value = kv[range(1, len(kv), 3)]
                feat_dict = dict(zip(key, value))
                common_feat_dict[line_list[0]] = feat_dict
                c += 1
        logger.info(f'get_common_dict n: {c:d}')
        return common_feat_dict


if __name__ == '__main__':
    logging.basicConfig(filename='./log/process_dataset.log', format="%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d %(message)s", level=logging.INFO)
    processer = process_dataset()
    processer.process('train')
    processer.process('test')
