from dataclasses import dataclass, field
import dataclasses
import logging
import utils
from typing import List, Union, Dict, Any, Mapping, Optional
import json
import torch
from model import AITM
from utils import optimizer_and_scheduler, loss_func
from dataprocessor import DataProcessor
from modelprocessor import ModelProcessor


logger = logging.getLogger(__name__)

@dataclass
class Config:
    batch_size: int = field(default=1000)
    embedding_size: int  = field(default=5)
    learning_rate: float = field(default=1e-4)
    epochs: int = field(default=10)
    earlystop_epoch: int = field(default=1)
    model_file: str = field(default='./out/AITM.model')
    vocab_dict: Mapping = field(default_factory= lambda : {'101': 238635, '121': 98, '122': 14, '124': 3, '125': 8, '126': 4, '127': 4, '128': 3, '129': 5, '205': 467298, '206': 6929, '207': 263942, '216': 106399, '508': 5888, '509': 104830, '702': 51878, '853': 37148, '301': 4})
    device: str = field(default='cuda' if torch.cuda.is_available() else 'cpu')
    weight_decay: float = field(default=1e-6)
    data_path: str = field(default='./data/ctr_cvr')
    shuffle: bool = field(default=True)
    logging_steps: int = field(default=1000)
    eval_steps: int = field(default=2000)
    output_dir: str = field(default='./output')

    def __repr__(self):
        self_asdict = dataclasses.asdict(self)
        return f'{self.__class__.__name__}\n' + json.dumps(self_asdict, indent=2) + '\n'

def train(config):
    # init dataloader
    dataprocessor = DataProcessor(config)
    # init model
    model = AITM(config.vocab_dict, config.embedding_size)
    logger.info(model)
    # init optimizer and scheduler
    optimizer, scheduler = optimizer_and_scheduler(config, model)
    # init model processor
    modelprocessor = ModelProcessor(config, model, dataprocessor, optimizer, scheduler, loss_func)
    # train dev test
    modelprocessor.process()

if __name__ == '__main__':
    logging.basicConfig(filename='./log/train.log', format="%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d %(message)s", level=logging.INFO)
    config = Config()
    logger.info(config)
    train(config)

