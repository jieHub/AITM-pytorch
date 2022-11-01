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
     
