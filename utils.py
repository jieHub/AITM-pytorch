import logging
import numpy as np
import torch
from torch import nn


logger = logging.getLogger(__name__)

def random_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_func(preds, labels, constraint_weight=0.6):
    click_preds, conversion_preds = preds

    click_loss = nn.functional.binary_cross_entropy(click_preds, labels[:, 0])
    conversion_loss = nn.functional.binary_cross_entropy(conversion_preds, labels[:, 1])
    constraint_loss = torch.sum(torch.maximum(conversion_preds - click_preds, torch.zeros_like(click_preds)))

    loss = click_loss + conversion_loss + constraint_weight * constraint_loss
    return loss

def optimizer_and_scheduler(config, model):
    optimizer = Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5)
    return optimizer, scheduler
     

def get_vocabulary_size(file_path):
    with open(file_path, 'r') as f:
        vocab_key = [key for key in f.readline().strip().split(',')]
        vocab_size = [0] * len(vocab_key)
        for idx, line in enumerate(f):
            if idx % 5000000 == 0: print(idx)
            values = line.strip().split(',')
            vocab_size = [max(vocab_size[i], int(values[i])) for i in range(len(values))]
        vocab_size = [x+1 for x in vocab_size]
    return dict(zip(vocab_key, vocab_size))


if __name__ == '__main__':
    dic = get_vocabulary_size('./data/ctr_cvr.train')
    print(dic)

