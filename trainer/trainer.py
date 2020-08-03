# -*- coding: utf-8 -*-
# @Time   : 2020/6/26 15:49
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : trainer.py

import os
import warnings
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from time import time
from trainer.utils import early_stopping, calculate_valid_score, dict2str
from evaluator import TopKEvaluator, LossEvaluator
from data.interaction import Interaction
from utils import ensure_dir, get_local_time


class AbstractTrainer(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    def __init__(self, config, model, logger):
        super(Trainer, self).__init__(config, model)

        self.logger = logger
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = config['eval_step']
        self.stopping_step = config['stopping_step']
        self.valid_metric = config['valid_metric']
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.checkpoint_dir = config['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        saved_model_file = '{}-{}.pth'.format(self.config['model'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -1
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        self.evaluator = TopKEvaluator(config, logger)

    def _build_optimizer(self):
        # todo: Avoid clear text strings
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        else:
            warnings.warn('Received unrecognized optimizer, set default Adam optimizer', UserWarning)
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data):
        self.model.train()
        total_loss = 0.
        for batch_idx, interaction in enumerate(train_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model.calculate_loss(interaction)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss

    def _valid_epoch(self, valid_data):
        valid_result = self.evaluate(valid_data, load_best_model=False)
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        return valid_score, valid_result

    def _save_checkpoint(self, epoch):
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, self.saved_model_file)

    def resume_checkpoint(self, resume_file):
        resume_file = str(resume_file)
        checkpoint = torch.load(resume_file)
        self.start_epoch = checkpoint['epoch'] + 1
        self.cur_step = checkpoint['cur_step']
        self.best_valid_score = checkpoint['best_valid_score']

        # load architecture params from checkpoint
        if checkpoint['config']['model'].lower() != self.config['model'].lower():
            warnings.warn('Architecture configuration given in config file is different from that of checkpoint. '
                          'This may yield an exception while state_dict is being loaded.', UserWarning)
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        print(message_output)

    def fit(self, train_data, valid_data=None, verbose=True):
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data)
            self.train_loss_dict[epoch_idx] = train_loss
            training_end_time = time()
            train_loss_output = "epoch %d training [time: %.2fs, train loss: %.4f]" % \
                                (epoch_idx, training_end_time - training_start_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)

            # eval
            if self.eval_step <= 0 or not valid_data:
                self._save_checkpoint(epoch_idx)
                update_output = 'Saving current: %s' % self.saved_model_file
                if verbose:
                    self.logger.info(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                if update_flag:
                    self._save_checkpoint(epoch_idx)
                    update_output = 'Saving current best: %s' % self.saved_model_file
                    self.best_valid_result = valid_result
                    if verbose:
                        self.logger.info(update_output)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        return self.best_valid_score, self.best_valid_result


    def evaluate(self, eval_data, load_best_model=True, model_file=None):
        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            #print(message_output)

        self.model.eval()
        batch_result_list, num_user_list = [], []
        for batch_idx, interaction in enumerate(eval_data):

            batch_size = interaction.length
            pos_len_list = interaction.pos_len_list   # type :list  number of positive item for each user in this batch
            user_idx_list = interaction.user_idx_list   # type :slice

            if batch_size <= self.test_batch_size:
                scores = self.model.predict(interaction.to(self.device))
            else:
                scores = self.spilt_predict(interaction, batch_size)

            batch_result = self.evaluator.evaluate(pos_len_list, scores, user_idx_list)
            batch_result_list.append(batch_result)
            num_user_list.append(len(pos_len_list))
        result = self.evaluator.collect(batch_result_list, num_user_list)

        return result

    def spilt_predict(self, interaction, batch_size):
        spilt_interaction = dict()
        for key, tensor in interaction.interaction.items():
            spilt_interaction[key] = tensor.split(self.test_batch_size, dim=0)
        num_block = (batch_size+self.test_batch_size-1)//self.test_batch_size
        result_list = []
        for i in range(num_block):
            current_interaction = dict()
            for key, spilt_tensor in spilt_interaction.items():
                current_interaction[key] = spilt_tensor[i]
            result = self.model.predict(Interaction(current_interaction).to(self.device))
            result_list.append(result)
        return torch.cat(result_list, dim=0)

    def plot_train_loss(self, show=True, save_path=None):
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)
