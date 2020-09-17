# @Time   : 2020/6/26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/7 18:38, 2020/9/15, 2020/8/21, 2020/8/19, 2020/9/16
# @Author : Zihan Lin, Yupeng Hou, Yushuo Chen, Shanlei Mu, Xingyu Pan
# @Email  : linzihan.super@foxmail.com, houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, slmu@ruc.edu.cn, panxy@ruc.edu.cn

import os
import itertools
from time import time
from logging import getLogger
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from recbox.evaluator import TopKEvaluator, LossEvaluator
from recbox.data.interaction import Interaction
from recbox.utils import ensure_dir, get_local_time, DataLoaderType, KGDataLoaderState, EvaluatorType, ModelType
from recbox.trainer.utils import early_stopping, calculate_valid_score, dict2str


def get_trainer(model_type):
    if model_type == ModelType.KNOWLEDGE:
        return KGTrainer
    else:
        return Trainer


class AbstractTrainer(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
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
        self.eval_type = config['eval_type']
        if self.eval_type == EvaluatorType.INDIVIDUAL:
            self.evaluator = LossEvaluator(config)
        else:
            self.evaluator = TopKEvaluator(config)

        self.item_tensor = None
        self.tot_item_num = None
        self.iid_field = config['ITEM_ID_FIELD']

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
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx):
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
            self.logger.warning('Architecture configuration given in config file is different from that of checkpoint. '
                                'This may yield an exception while state_dict is being loaded.')
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        print(message_output)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True):
        if hasattr(self.model, 'train_preparation'):
            self.model.train_preparation(train_data=train_data, valid_data=valid_data)
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx)
            self.train_loss_dict[epoch_idx] = train_loss
            training_end_time = time()
            train_loss_output = "epoch %d training [time: %.2fs, train loss: %.4f]" % \
                                (epoch_idx, training_end_time - training_start_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
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
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = 'Saving current best: %s' % self.saved_model_file
                        if verbose:
                            self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        return self.best_valid_score, self.best_valid_result

    def _full_sort_batch_eval(self, batched_data):
        # Note: interaction without item ids
        interaction, pos_idx, used_idx, pos_len_list, neg_len_list = batched_data

        batch_size = interaction.length * self.tot_item_num
        if hasattr(self.model, 'full_sort_predict'):
            # Note: interaction without item ids
            scores = self.model.full_sort_predict(interaction.to(self.device)).flatten()
        else:
            interaction = interaction.to(self.device).repeat_interleave(self.tot_item_num)
            interaction.update(self.item_tensor[:batch_size])
            scores = self.model.predict(interaction)
        pos_idx = pos_idx.to(self.device)
        used_idx = used_idx.to(self.device)

        pos_scores = scores.index_select(dim=0, index=pos_idx)
        pos_scores = torch.split(pos_scores, pos_len_list, dim=0)

        ones_tensor = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        used_mask = ones_tensor.index_fill(dim=0, index=used_idx, value=0)
        neg_scores = scores.masked_select(used_mask)
        neg_scores = torch.split(neg_scores, neg_len_list, dim=0)

        final_scores = list(itertools.chain.from_iterable(zip(pos_scores, neg_scores)))
        final_scores = torch.cat(final_scores)

        setattr(interaction, 'pos_len_list', pos_len_list)
        setattr(interaction, 'user_len_list', list(np.add(pos_len_list, neg_len_list)))

        return interaction, final_scores

    @torch.no_grad()
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

        if eval_data.dl_type == DataLoaderType.FULL:
            if not hasattr(self.model, 'full_sort_predict'):
                self.item_tensor = eval_data.get_item_feature().to(self.device).repeat(eval_data.step)
            self.tot_item_num = eval_data.dataset.item_num

        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            if eval_data.dl_type == DataLoaderType.FULL:
                if self.eval_type == EvaluatorType.INDIVIDUAL:
                    raise ValueError('full sort can\'t use LossEvaluator')
                interaction, scores = self._full_sort_batch_eval(batched_data)
            else:
                interaction = batched_data
                batch_size = interaction.length

                if batch_size <= self.test_batch_size:
                    scores = self.model.predict(interaction.to(self.device))
                else:
                    scores = self.spilt_predict(interaction, batch_size)

            batch_matrix = self.evaluator.evaluate(interaction, scores)
            batch_matrix_list.append(batch_matrix)
        result = self.evaluator.collect(batch_matrix_list, eval_data)

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


class KGTrainer(Trainer):
    def __init__(self, config, model):
        super(KGTrainer, self).__init__(config, model)

        self.train_rec_step = config['train_rec_step']
        self.train_kg_step = config['train_kg_step']

    def _train_epoch(self, train_data, epoch_idx):
        self.model.train()
        total_loss = 0.
        if self.train_rec_step is None or self.train_kg_step is None:
            interaction_state = KGDataLoaderState.RSKG
        else:
            assert self.train_rec_step > 0 and self.train_kg_step > 0
            interaction_state = KGDataLoaderState.RS \
                if epoch_idx % (self.train_rec_step + self.train_kg_step) < self.train_rec_step \
                else KGDataLoaderState.KG
        train_data.set_mode(interaction_state)
        if interaction_state in [KGDataLoaderState.RSKG, KGDataLoaderState.RS]:
            for batch_idx, interaction in enumerate(train_data):
                interaction = interaction.to(self.device)
                self.optimizer.zero_grad()
                loss = self.model.calculate_loss(interaction)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        elif interaction_state in [KGDataLoaderState.KG]:
            for bath_idx, interaction in enumerate(train_data):
                interaction = interaction.to(self.device)
                self.optimizer.zero_grad()
                loss = self.model.calculate_kg_loss(interaction)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        return total_loss
