# @Time   : 2021/6/23
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

"""
recbole.evaluator.collector
################################################
"""

from recbole.evaluator.register import Register
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import copy

class DataStruct(object):
    def __init__(self):
        self._data_dict = {}

    def __getitem__(self, name: str):
        return self._data_dict[name]

    def __setitem__(self, name: str, value):
        self._data_dict[name] = value

    def __delitem__(self, name: str):
        self._data_dict.pop(name)

    def __contains__(self, key: str):
        return key in self._data_dict

    def get(self, name: str):
        if name not in self._data_dict:
            raise IndexError("Can not load the data without registration !")
        return self[name]

    def set(self, name: str, value):
        self._data_dict[name] = value

    def update_tensor(self, name: str, value: torch.Tensor):
        if name not in self._data_dict:
            self._data_dict[name] = value.cpu().clone().detach()
        else:
            if not isinstance(self._data_dict[name], torch.Tensor):
                raise ValueError("{} is not a tensor.".format(name))
            self._data_dict[name] = torch.cat((self._data_dict[name], value.cpu().clone().detach()), dim=0)

    def __str__(self):
        data_info = '\nContaining:\n'
        for data_key in self._data_dict.keys():
            data_info += data_key + '\n'
        return data_info


class Collector(object):
    """The collector is used to collect the resource for evaluator.
        As the evaluation metrics are various, the needed resource not only contain the recommended result
        but also other resource from data and model. They all can be collected by the collector during the training
        and evaluation process.

        This class is only used in Trainer.

    """
    def __init__(self, config):
        self.config = config
        self.data_struct = DataStruct()
        self.register = Register(config)
        self.full = ('full' in config['eval_setting'])
        self.topk = self.config['topk']
        self.topk_idx = None

    def data_collect(self, train_data):
        """ Collect the evaluation resource from training data.
            Args:
                train_data (AbstractDataLoader): the training dataloader which contains the training data.

        """
        if self.register.need('data.num_item'):
            self.data_struct.set('data.num_item', train_data.num_item)
        if self.register.need('data.num_user'):
            self.data_struct.set('data.num_user', train_data.num_user)

    def _get_score_matrix(self, scores_tensor, user_len_list):
        """get score matrix.

            Args:
                scores_tensor (tensor): the tensor of model output with size of `(N, )`
                user_len_list(list): number of all items

        """
        if self.full:
            scores_matrix = scores_tensor.reshape(len(user_len_list), -1)
        else:
            scores_list = torch.split(scores_tensor, user_len_list, dim=0)
            if scores_tensor.dtype is torch.FloatTensor:
                scores_matrix = pad_sequence(scores_list, batch_first=True, padding_value=-np.inf)  # n_users x items
            else:  # padding the id tensor
                scores_matrix = pad_sequence(scores_list, batch_first=True, padding_value=-1)  # n_users x items
        return scores_matrix

    def _average_rank(self, scores):
        """Get the ranking of an ordered tensor, and take the average of the ranking for positions with equal values.

        Args:
            scores(tensor): an ordered tensor, with size of `(N, )`

        Returns:
            torch.Tensor: average_rank

        Example:
            >>> average_rank(tensor([[1,2,2,2,3,3,6],[2,2,2,2,4,5,5]]))
            tensor([[1.0000, 3.0000, 3.0000, 3.0000, 5.5000, 5.5000, 7.0000],
            [2.5000, 2.5000, 2.5000, 2.5000, 5.0000, 6.5000, 6.5000]])

        Reference:
            https://github.com/scipy/scipy/blob/v0.17.1/scipy/stats/stats.py#L5262-L5352

        """
        length, width = scores.shape
        device = scores.device
        true_tensor = torch.full((length, 1), True, dtype=torch.bool, device=device)

        obs = torch.cat([true_tensor, scores[:, 1:] != scores[:, :-1]], dim=1)
        # bias added to dense
        bias = torch.arange(0, length, device=device).repeat(width).reshape(width, -1). \
            transpose(1, 0).reshape(-1)
        dense = obs.view(-1).cumsum(0) + bias

        # cumulative counts of each unique value
        count = torch.where(torch.cat([obs, true_tensor], dim=1))[1]
        # get average rank
        avg_rank = .5 * (count[dense] + count[dense - 1] + 1).view(length, -1)

        return avg_rank

    def eval_batch_collect(self, scores_tensor: torch.Tensor, interaction):
        """ Collect the evaluation resource from batched eval data and batched model output.
            Args:
                scores_tensor (Torch.Tensor): the output tensor of model with the shape of `(N, )`
                interaction(Interaction): batched eval data.
        """
        if self.register.need('rec.topk'):

            user_len_list = interaction.user_len_list
            pos_len_list = interaction.pos_len_list

            scores_matrix = self._get_score_matrix(scores_tensor, user_len_list)
            scores_matrix = torch.flip(scores_matrix, dims=[-1])
            shape_matrix = torch.full((len(user_len_list), 1), scores_matrix.shape[1], device=scores_matrix.device)

            pos_len_matrix = torch.from_numpy(np.array(pos_len_list)).view(-1, 1).to(scores_matrix.device)

            assert pos_len_matrix.shape[0] == shape_matrix.shape[0]

            # get topk
            _, topk_idx = torch.topk(scores_matrix, max(self.topk), dim=-1)  # n_users x k
            self.topk_idx = topk_idx
            # pack top_idx and shape_matrix
            result = torch.cat((topk_idx, shape_matrix, pos_len_matrix), dim=1)
            self.data_struct.update_tensor('rec.topk', result)

        if self.register.need('rec.meanrank'):

            user_len_list = interaction.user_len_list
            pos_len_list = interaction.pos_len_list
            pos_len_tensor = torch.Tensor(pos_len_list).to(scores_tensor.device)
            scores_matrix = self._get_score_matrix(scores_tensor, user_len_list)
            desc_scores, desc_index = torch.sort(scores_matrix, dim=-1, descending=True)

            # get the index of positive items in the ranking list
            pos_index = (desc_index < pos_len_tensor.reshape(-1, 1))

            avg_rank = self._average_rank(desc_scores)
            pos_rank_sum = torch.where(pos_index, avg_rank, torch.zeros_like(avg_rank)).sum(axis=-1).reshape(-1, 1)

            pos_len_matrix = torch.from_numpy(np.array(pos_len_list)).view(-1, 1).to(scores_matrix.device)
            user_len_matrix = torch.from_numpy(np.array(user_len_list)).view(-1, 1).to(scores_matrix.device)

            result = torch.cat((pos_rank_sum, user_len_matrix, pos_len_matrix), dim=1)
            self.data_struct.update_tensor('rec.meanrank', result)

        if self.register.need('rec.score'):

            self.data_struct.update_tensor('rec.score', scores_tensor)

        if self.register.need('data.label'):
            self.label_field = self.config['LABEL_FIELD']
            self.data_struct.update_tensor('data.label', interaction[self.label_field].to(scores_tensor.device))

        if self.register.need('rec.items'):
            if not self.register.need('rec.topk'):
                raise ValueError("Recommended items is only prepared for top-k metrics!")
            if self.full:
                self.data_struct.update_tensor('rec.items', self.topk_idx)
            else:
                self.item_field = self.config['ITEM_ID_FIELD']
                user_len_list = interaction.user_len_list
                item_tensor = interaction[self.item_field].to(scores_tensor.device)
                item_matrix = self._get_score_matrix(item_tensor, user_len_list)  # n_user * n_items
                topk_item = torch.gather(item_matrix, dim=1, index=self.topk_idx)  # n_user * k

                self.data_struct.update_tensor('rec.items', topk_item)

    def model_collect(self, model: torch.nn.Module):

        """ Collect the evaluation resource from model.
            Args:
                model (nn.Module): the trained recommendation model.
        """
        pass
        # TODO:

    def eval_collect(self, eval_pred: torch.Tensor, data_label: torch.Tensor):
        """ Collect the evaluation resource from total output and label.
            It was designed for those models that can not predict with batch.
            Args:
                eval_pred (torch.Tensor): the output score tensor of model.
                data_label (torch.Tensor): the label tensor.
        """
        if self.register.need('rec.score'):
            self.data_struct.update_tensor('rec.score', eval_pred)

        if self.register.need('data.label'):
            self.label_field = self.config['LABEL_FIELD']
            self.data_struct.update_tensor('data.label', data_label.to(eval_pred.device))

    def get_data_struct(self):
        """ Get all the evaluation resource that been collected.
            And reset some of outdated resource.
        """
        returned_struct = copy.deepcopy(self.data_struct)
        for key in ['rec.topk', 'rec.meanrank', 'rec.score', 'rec.items', 'data.label']:
            if key in self.data_struct:
                del self.data_struct[key]
        return returned_struct
