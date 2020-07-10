# -*- coding: utf-8 -*-
# @Time   : 2020/6/26 15:49
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : trainer.py

import warnings
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from time import time
from trainer.utils import early_stopping, calculate_valid_score, dict2str
from evaluator import Evaluator


class Trainer(object):
    def __init__(self, config, logger, model):
        self.config = config
        self.logger = logger
        self.learner = config['model.learner']
        self.learning_rate = config['model.learning_rate']
        self.epochs = config['train.epochs']
        self.eval_step = config['model.eval_step']
        self.stopping_step = config['model.stopping_step']
        self.device = config['device']
        self.checkpoint_dir = config['model.checkpoint_dir']

        self.start_epoch = 0
        self.cur_step = 0
        self.best_eval_score = -1
        self.train_loss_dict = dict()
        self.model = model
        self.optimizer = self._build_optimizer()
        self.evaluator = Evaluator(config, logger)

    def _build_optimizer(self):
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
            loss = self.model.train_model(interaction)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss

    def _valid_epoch(self, valid_data):
        self.model.eval()
        valid_result = self.evaluate(valid_data)
        valid_score = calculate_valid_score(valid_result)
        return valid_score, valid_result

    def _save_checkpoint(self, epoch):
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_eval_score': self.best_eval_score,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        filename = str(self.checkpoint_dir + '/model_best.pth')
        torch.save(state, filename)

    def resume_checkpoint(self, resume_file):
        resume_file = str(resume_file)
        checkpoint = torch.load(resume_file)
        self.start_epoch = checkpoint['epoch'] + 1
        self.cur_step = checkpoint['cur_step']
        self.best_eval_score = checkpoint['best_eval_score']

        # load architecture params from checkpoint
        if checkpoint['config']['model'].lower() != self.config['model'].lower():
            warnings.warn('Architecture configuration given in config file is different from that of checkpoint. '
                          'This may yield an exception while state_dict is being loaded.', UserWarning)
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        print(message_output)

    def evaluate(self, eval_data):
        batch_result_list, batch_size_list = [], []
        for batch_idx, interaction in enumerate(eval_data):
            interaction = interaction.to(self.device)
            # todo:
            USER_ID = self.config['data.USER_ID_FIELD']
            ITEM_ID = self.config['data.ITEM_ID_FIELD']
            users, items = interaction[USER_ID], interaction[ITEM_ID]
            scores = self.model.predict(interaction)
            batch_size = users.size()[0]
            users = users.cpu().numpy()
            items = items.cpu().numpy()
            scores = scores.detach().cpu().numpy()
            batch_result = self.evaluator.evaluate([users, items, scores], eval_data)
            batch_result_list.append(batch_result)
            batch_size_list.append(batch_size)
        result = self.evaluator.collect(batch_result_list, batch_size_list)
        return result

    def train(self, train_data, valid_data):
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data)
            self.train_loss_dict[epoch_idx] = train_loss
            training_end_time = time()
            train_loss_output = "epoch %d training [time: %.2fs, train loss: %.4f]" % \
                                (epoch_idx, training_end_time - training_start_time, train_loss)
            print(train_loss_output)

            # eval
            if self.eval_step <= 0:
                self._save_checkpoint(epoch_idx)
                update_output = 'Saving current: model_best.pth'
                print(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.best_eval_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_eval_score, self.cur_step, max_step=self.stopping_step, order='asc')
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                print(valid_score_output)
                print(valid_result_output)
                if update_flag:
                    self._save_checkpoint(epoch_idx)
                    update_output = 'Saving current best: model_best.pth'
                    print(update_output)
                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    print(stop_output)
                    break

    def test(self, test_data, load_best_model=True):
        if load_best_model:
            # todo: more flexible settings
            checkpoint_file = self.checkpoint_dir + '/model_best.pth'
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            print(message_output)
        self.model.eval()
        test_result = self.evaluate(test_data)
        return test_result

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
