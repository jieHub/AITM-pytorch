import os
import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)


class ModelProcessor():

    def __init__(self, config, model, data_processor, optimizer, lr_scheduler):
        super().__init__()
        self.config = config
        self.model = model
        self.data_processor = data_processor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def process(self):
        logger.info('train model...')
        self._train_model('train')
        logger.info('evaluate test set...')
        self.model.load_state_dict(torch.load(self.save_path, map_location=self.config.device))
        loss, metric = self._evaluate_model()
        logger.info(f'Evaluate Testset: Loss {loss} - Metric {metric}')
        self.save_path = os.path.join(self.config.output_dir, f'checkpoint-best.pt')
        torch.save(self.model.state_dict(), self.save_path)
        logger.info(f'train model end; best model save in {self.save_path}')

    def _train_model(self, flag):
        train_dataloader = self.data_processor.process(flag)
        num_update_steps_per_epoch = max(len(train_dataloader), 1)

        self.model.to(self.config.device)
        self.model.train()

        best_metric, best_step = 0, -1
        global_step = 0
        logger.info('******start training model ...******')
        logger.inof(f'   Num sample = {num_sample}')
        logger.info(f'   Num epochs = {self.config.epochs}')
        logger.info(f'   Batch size = {self.config.batchsize}')

        for epoch in range(self.config.epochs):
            for step, (features, labels) in enumerate(train_dataloader):
                features, labels = self._prepare_input(features, self.config.device), self._prepare_input(labels, self.config.device)
                outputs = self.model(features)
                loss = self.loss_func(outputs, labels)

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step(epoch + step / num_update_steps_per_epoch)
                global_step += 1

                if global_step % self.config.logging_steps == 0:
                    logger.info(f'Training: Epoch {epoch+1}/{self.config.epochs} - Step {step + 1} - Loss {loss}')
                if global_step % self.config.eval_steps == 0:
                    loss, metric = self._evaluate_model(flag)
                    logger.info(f'Evaluate: Epoch {epoch+1}/{self.config.epochs} - GlobalStep {global_step + 1} - Loss {loss} - Metric {metric}')
                    if metric > best_metric:
                        best_metric, best_step = metric, global_step
                        self.save_path = os.path.join(self.config.output_dir, f'checkpoint-{best_step}.pt')
                        torch.save(self.model.state_dict(), self.save_path)
        return best_step, best_metric

    def _evaluate_model(self, flag):













    def _prepare_input(self, data, device='cuda'):
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v, device) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v, device) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        return data
