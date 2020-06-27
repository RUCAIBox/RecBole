# -*- coding: utf-8 -*-
# @Time   : 2020/6/26 15:49
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : trainer.py

import warnings
import torch
import torch.optim as optim

from time import time
from trainer.utils import early_stopping


class Trainer(object):
    def __init__(self, model, config):
        self.config = config
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.eval_step = config['eval_step']
        self.stopping_step = config['stopping_step']
        self.device = config['device']
        self.checkpoint_dir = config['checkpoint_dir']

        self.start_epoch = 0
        self.cur_step = 0
        self.best_eval_score = -1
        self.model = model
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self):
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            warnings.warn('Received unrecognized optimizer, set default Adam optimizer', UserWarning)
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data):
        self.model.train()

        # for test
        # start = 0
        # total_loss = 0.
        # n_train = train_data.shape[0]
        # while start + self.batch_size <= n_train:
        #     end = start + self.batch_size
        #     users = train_data[start:end, 0]
        #     pos_items = train_data[start:end, 1]
        #     neg_items = train_data[start:end, 2]
        #     users = torch.from_numpy(users).to(self.device)
        #     pos_items = torch.from_numpy(pos_items).to(self.device)
        #     neg_items = torch.from_numpy(neg_items).to(self.device)
        #     self.optimizer.zero_grad()
        #     loss = self.model.train_model(users, pos_items, neg_items)
        #     loss.backward()
        #     self.optimizer.step()
        #     start += self.batch_size
        #     total_loss += loss.item()
        # return total_loss

        total_loss = 0.
        for batch_idx, (users, pos_items, neg_items) in enumerate(train_data):
            self.optimizer.zero_grad()
            loss = self.model.train_model(users, pos_items, neg_items)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss

    def _valid_epoch(self, valid_data):
        # self.model.eval()
        # for batch_idx, (users, items, _) in enumerate(valid_data):
        #     item_scores = self.model.predict(users, items)
        evaluator_score = 0
        return evaluator_score

    def _save_checkpoint(self, epoch):
        arch = type(self.model).__name__
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
        if checkpoint['config']['recommender'].lower() != self.config['recommender'].lower():
            warnings.warn('Architecture configuration given in config file is different from that of checkpoint. '
                          'This may yield an exception while state_dict is being loaded.', UserWarning)
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        print(message_output)

    def train(self, train_data, eval_data):

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data)
            training_end_time = time()
            train_loss_output = "epoch %d training [time: %.2fs, train loss: %.4f]" % \
                                (epoch_idx, training_end_time - training_start_time, train_loss)
            print(train_loss_output)

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                eval_start_time = time()
                eval_score = self._valid_epoch(eval_data)
                self.best_eval_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    eval_score, self.best_eval_score, self.cur_step, max_step=self.stopping_step, order='asc')
                eval_end_time = time()
                eval_score_output = "epoch %d evaluating [time: %.2fs, eval_score: ]" % \
                                    (epoch_idx, eval_end_time - eval_start_time)
                print(eval_score_output)
                if update_flag:
                    self._save_checkpoint(epoch_idx)
                    update_output = 'Saving current best: mode_best.pth'
                    print(update_output)
                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    print(stop_output)
                    break
