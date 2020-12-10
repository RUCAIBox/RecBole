# @Time   : 2020/6/26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/7, 2020/9/26, 2020/9/26, 2020/10/01, 2020/9/16, 2020/10/8, 2020/10/15
# @Author : Zihan Lin, Yupeng Hou, Yushuo Chen, Shanlei Mu, Xingyu Pan, Hui Wang, Xinyan Fan
# @Email  : linzihan.super@foxmail.com, houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, slmu@ruc.edu.cn, panxy@ruc.edu.cn, hui.wang@ruc.edu.cn, xinyan.fan@ruc.edu.cn

r"""
recbole.trainer.trainer
################################
"""

import os
import itertools
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt

from time import time
from logging import getLogger

from recbole.evaluator import TopKEvaluator, LossEvaluator
from recbole.data.interaction import Interaction
from recbole.utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str, \
    DataLoaderType, KGDataLoaderState, EvaluatorType


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
    resume_checkpoint() and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
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

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
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

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        for batch_idx, interaction in enumerate(train_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
        return total_loss

    def _valid_epoch(self, valid_data):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data, load_best_model=False)
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        return valid_score, valid_result

    def _save_checkpoint(self, epoch):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
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
        r"""Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file

        """
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
        self.logger.info(message_output)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, verbose=True, saved=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1)

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
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
        used_idx = torch.cat([used_idx, torch.arange(interaction.length) * self.tot_item_num])  # remove [pad] item
        neg_len_list = list(np.subtract(neg_len_list, 1))
        try:
            # Note: interaction without item ids
            scores = self.model.full_sort_predict(interaction.to(self.device)).flatten()
        except NotImplementedError:
            interaction = interaction.to(self.device).repeat_interleave(self.tot_item_num)
            interaction.update(self.item_tensor[:batch_size])
            if batch_size <= self.test_batch_size:
                scores = self.model.predict(interaction)
            else:
                scores = self._spilt_predict(interaction, batch_size)
        pos_idx = pos_idx.to(self.device)
        used_idx = used_idx.to(self.device)

        pos_scores = scores.index_select(dim=0, index=pos_idx)
        pos_scores = torch.split(pos_scores, pos_len_list, dim=0)

        ones_tensor = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        used_mask = ones_tensor.index_fill(dim=0, index=used_idx, value=0)
        neg_scores = scores.masked_select(used_mask)
        neg_scores = torch.split(neg_scores, neg_len_list, dim=0)

        tmp_len_list = np.add(pos_len_list, neg_len_list).tolist()
        final_scores_width = max(self.tot_item_num, max(tmp_len_list))
        extra_len_list = np.subtract(final_scores_width, tmp_len_list).tolist()
        padding_nums = final_scores_width * len(tmp_len_list) - np.sum(tmp_len_list)
        padding_tensor = torch.tensor([-np.inf], dtype=scores.dtype, device=self.device).repeat(padding_nums)
        padding_scores = torch.split(padding_tensor, extra_len_list)

        final_scores = list(itertools.chain.from_iterable(zip(pos_scores, neg_scores, padding_scores)))
        final_scores = torch.cat(final_scores)

        setattr(interaction, 'pos_len_list', pos_len_list)
        setattr(interaction, 'user_len_list', len(tmp_len_list) * [final_scores_width])

        return interaction, final_scores

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.

        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        if eval_data.dl_type == DataLoaderType.FULL:
            if self.item_tensor is None:
                self.item_tensor = eval_data.get_item_feature().to(self.device).repeat(eval_data.step)
            self.tot_item_num = eval_data.dataset.item_num

        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            if eval_data.dl_type == DataLoaderType.FULL:
                if self.eval_type == EvaluatorType.INDIVIDUAL:
                    raise ValueError('full sort can\'t use LossEvaluator')
                interaction, scores = self._full_sort_batch_eval(batched_data)
                batch_matrix = self.evaluator.collect(interaction, scores, full=True)
            else:
                interaction = batched_data
                batch_size = interaction.length

                if batch_size <= self.test_batch_size:
                    scores = self.model.predict(interaction.to(self.device))
                else:
                    scores = self._spilt_predict(interaction, batch_size)

                batch_matrix = self.evaluator.collect(interaction, scores)
            batch_matrix_list.append(batch_matrix)
        result = self.evaluator.evaluate(batch_matrix_list, eval_data)

        return result

    def _spilt_predict(self, interaction, batch_size):
        spilt_interaction = dict()
        for key, tensor in interaction.interaction.items():
            spilt_interaction[key] = tensor.split(self.test_batch_size, dim=0)
        num_block = (batch_size + self.test_batch_size - 1) // self.test_batch_size
        result_list = []
        for i in range(num_block):
            current_interaction = dict()
            for key, spilt_tensor in spilt_interaction.items():
                current_interaction[key] = spilt_tensor[i]
            result = self.model.predict(Interaction(current_interaction).to(self.device))
            if len(result.shape) == 0:
                result = result.unsqueeze(0)
            result_list.append(result)
        return torch.cat(result_list, dim=0)

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
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
    r"""KGTrainer is designed for Knowledge-aware recommendation methods. Some of these models need to train the
    recommendation related task and knowledge related task alternately.

    """

    def __init__(self, config, model):
        super(KGTrainer, self).__init__(config, model)

        self.train_rec_step = config['train_rec_step']
        self.train_kg_step = config['train_kg_step']

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        if self.train_rec_step is None or self.train_kg_step is None:
            interaction_state = KGDataLoaderState.RSKG
        elif epoch_idx % (self.train_rec_step + self.train_kg_step) < self.train_rec_step:
            interaction_state = KGDataLoaderState.RS
        else:
            interaction_state = KGDataLoaderState.KG
        train_data.set_mode(interaction_state)
        if interaction_state in [KGDataLoaderState.RSKG, KGDataLoaderState.RS]:
            return super()._train_epoch(train_data, epoch_idx)
        elif interaction_state in [KGDataLoaderState.KG]:
            return super()._train_epoch(train_data, epoch_idx, self.model.calculate_kg_loss)
        return None


class KGATTrainer(Trainer):
    r"""KGATTrainer is designed for KGAT, which is a knowledge-aware recommendation method.

    """

    def __init__(self, config, model):
        super(KGATTrainer, self).__init__(config, model)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        # train rs
        train_data.set_mode(KGDataLoaderState.RS)
        rs_total_loss = super()._train_epoch(train_data, epoch_idx)

        # train kg
        train_data.set_mode(KGDataLoaderState.KG)
        kg_total_loss = super()._train_epoch(train_data, epoch_idx, self.model.calculate_kg_loss)

        # update A
        self.model.update_attentive_A()

        return rs_total_loss, kg_total_loss


class S3RecTrainer(Trainer):
    r"""S3RecTrainer is designed for S3Rec, which is a self-supervised learning based sequential recommenders.
        It includes two training stages: pre-training ang fine-tuning.

        """

    def __init__(self, config, model):
        super(S3RecTrainer, self).__init__(config, model)

    def save_pretrained_model(self, epoch, saved_model_file):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id
            saved_model_file (str): file name for saved pretrained model

        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, saved_model_file)

    def pretrain(self, train_data, verbose=True):

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)

            if (epoch_idx + 1) % self.config['save_step'] == 0:
                saved_model_file = os.path.join(self.checkpoint_dir,
                                                '{}-{}-{}.pth'.format(self.config['model'], self.config['dataset'],
                                                                      str(epoch_idx + 1)))
                self.save_pretrained_model(epoch_idx, saved_model_file)
                update_output = 'Saving current: %s' % saved_model_file
                if verbose:
                    self.logger.info(update_output)

        return self.best_valid_score, self.best_valid_result

    def fit(self, train_data, valid_data=None, verbose=True, saved=True):
        if self.model.train_stage == 'pretrain':
            return self.pretrain(train_data, verbose)
        elif self.model.train_stage == 'finetune':
            return super().fit(train_data, valid_data, verbose, saved)
        else:
            raise ValueError("Please make sure that the 'train_stage' is 'pretrain' or 'finetune' ")


class MKRTrainer(Trainer):
    r"""MKRTrainer is designed for MKR, which is a knowledge-aware recommendation method.

    """

    def __init__(self, config, model):
        super(MKRTrainer, self).__init__(config, model)
        self.kge_interval = config['kge_interval']

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        rs_total_loss, kg_total_loss = 0., 0.

        # train rs
        self.logger.info('Train RS')
        train_data.set_mode(KGDataLoaderState.RS)
        rs_total_loss = super()._train_epoch(train_data, epoch_idx, self.model.calculate_rs_loss)

        # train kg
        if epoch_idx % self.kge_interval == 0:
            self.logger.info('Train KG')
            train_data.set_mode(KGDataLoaderState.KG)
            kg_total_loss = super()._train_epoch(train_data, epoch_idx, self.model.calculate_kg_loss)

        return rs_total_loss, kg_total_loss


class TraditionalTrainer(Trainer):
    r"""TraditionalTrainer is designed for Traditional model(Pop,ItemKNN), which set the epoch to 1 whatever the config.

    """

    def __init__(self, config, model):
        super(TraditionalTrainer, self).__init__(config, model)
        self.epochs = 1  # Set the epoch to 1 when running memory based model
