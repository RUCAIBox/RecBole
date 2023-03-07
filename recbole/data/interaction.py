# @Time   : 2020/7/10
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time    : 2022/7/8, 2020/9/15, 2020/9/16, 2020/8/12
# @Author  : Zhen Tian, Yupeng Hou, Yushuo Chen, Xingyu Pan
# @email   : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, panxy@ruc.edu.cn

"""
recbole.data.interaction
############################
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn_utils


def _convert_to_tensor(data):
    """This function can convert common data types (list, pandas.Series, numpy.ndarray, torch.Tensor) into torch.Tensor.

    Args:
        data (list, pandas.Series, numpy.ndarray, torch.Tensor): Origin data.

    Returns:
        torch.Tensor: Converted tensor from `data`.
    """
    elem = data[0]
    if isinstance(elem, (float, int, np.float, np.int64)):
        new_data = torch.as_tensor(data)
    elif isinstance(elem, (list, tuple, pd.Series, np.ndarray, torch.Tensor)):
        seq_data = [torch.as_tensor(d) for d in data]
        new_data = rnn_utils.pad_sequence(seq_data, batch_first=True)
    else:
        raise ValueError(f"[{type(elem)}] is not supported!")
    if new_data.dtype == torch.float64:
        new_data = new_data.float()
    return new_data


class Interaction(object):
    """The basic class representing a batch of interaction records.

    Note:
        While training, there is no strict rules for data in one Interaction object.

        While testing, it should be guaranteed that all interaction records of one single
        user will not appear in different Interaction object, and records of the same user
        should be continuous. Meanwhile, the positive cases of one user always need to occur
        **earlier** than this user's negative cases.

        A correct example:
            =======     =======     =======
            user_id     item_id     label
            =======     =======     =======
            1           2           1
            1           6           1
            1           3           1
            1           1           0
            2           3           1
            ...         ...         ...
            =======     =======     =======

        Some wrong examples for Interaction objects used in testing:

        1.
            =======     =======     =======     ============
            user_id     item_id     label
            =======     =======     =======     ============
            1           2           1
            1           6           0           # positive cases of one user always need to

                                                occur earlier than this user's negative cases
            1           3           1
            1           1           0
            2           3           1
            ...         ...         ...
            =======     =======     =======     ============

        2.
            =======     =======     =======     ========
            user_id     item_id     label
            =======     =======     =======     ========
            1           2           1
            1           6           1
            1           3           1
            2           3           1           # records of the same user should be continuous.
            1           1           0
            ...         ...         ...
            =======     =======     =======     ========

    Attributes:
        interaction (dict or pandas.DataFrame): keys are meaningful str (also can be called field name),
            and values are Torch Tensor of numpy Array with shape (batch_size, \\*).
    """

    def __init__(self, interaction):
        self.interaction = dict()
        if isinstance(interaction, dict):
            for key, value in interaction.items():
                if isinstance(value, (list, np.ndarray)):
                    self.interaction[key] = _convert_to_tensor(value)
                elif isinstance(value, torch.Tensor):
                    self.interaction[key] = value
                else:
                    raise ValueError(
                        f"The type of {key}[{type(value)}] is not supported!"
                    )
        elif isinstance(interaction, pd.DataFrame):
            for key in interaction:
                value = interaction[key].values
                self.interaction[key] = _convert_to_tensor(value)
        else:
            raise ValueError(
                f"[{type(interaction)}] is not supported for initialize `Interaction`!"
            )
        self.length = -1
        for k in self.interaction:
            self.length = max(self.length, self.interaction[k].unsqueeze(-1).shape[0])

    def __iter__(self):
        return self.interaction.__iter__()

    def __getattr__(self, item):
        if "interaction" not in self.__dict__:
            raise AttributeError(f"'Interaction' object has no attribute 'interaction'")
        if item in self.interaction:
            return self.interaction[item]
        raise AttributeError(f"'Interaction' object has no attribute '{item}'")

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.interaction[index]
        if isinstance(index, (np.ndarray, torch.Tensor)):
            index = index.tolist()

        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k][index]
        return Interaction(ret)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise KeyError(f"{type(key)} object does not support item assigment")
        self.interaction[key] = value

    def __delitem__(self, key):
        if key not in self.interaction:
            raise KeyError(f"{type(key)} object does not in this interaction")
        del self.interaction[key]

    def __contains__(self, item):
        return item in self.interaction

    def __len__(self):
        return self.length

    def __str__(self):
        info = [f"The batch_size of interaction: {self.length}"]
        for k in self.interaction:
            inter = self.interaction[k]
            temp_str = f"    {k}, {inter.shape}, {inter.device.type}, {inter.dtype}"
            info.append(temp_str)
        info.append("\n")
        return "\n".join(info)

    def __repr__(self):
        return self.__str__()

    @property
    def columns(self):
        """
        Returns:
            list of str: The columns of interaction.
        """
        return list(self.interaction.keys())

    def to(self, device, selected_field=None):
        """Transfer Tensors in this Interaction object to the specified device.

        Args:
            device (torch.device): target device.
            selected_field (str or iterable object, optional): if specified, only Tensors
            with keys in selected_field will be sent to device.

        Returns:
            Interaction: a coped Interaction object with Tensors which are sent to
            the specified device.
        """
        ret = {}
        if isinstance(selected_field, str):
            selected_field = [selected_field]

        if selected_field is not None:
            selected_field = set(selected_field)
            for k in self.interaction:
                if k in selected_field:
                    ret[k] = self.interaction[k].to(device)
                else:
                    ret[k] = self.interaction[k]
        else:
            for k in self.interaction:
                ret[k] = self.interaction[k].to(device)
        return Interaction(ret)

    def cpu(self):
        """Transfer Tensors in this Interaction object to cpu.

        Returns:
            Interaction: a coped Interaction object with Tensors which are sent to cpu.
        """
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].cpu()
        return Interaction(ret)

    def numpy(self):
        """Transfer Tensors to numpy arrays.

        Returns:
            dict: keys the same as Interaction object, are values are corresponding numpy
            arrays transformed from Tensor.
        """
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].numpy()
        return ret

    def repeat(self, sizes):
        """Repeats each tensor along the batch dim.

        Args:
            sizes (int): repeat times.

        Example:
            >>> a = Interaction({'k': torch.zeros(4)})
            >>> a.repeat(3)
            The batch_size of interaction: 12
                k, torch.Size([12]), cpu

            >>> a = Interaction({'k': torch.zeros(4, 7)})
            >>> a.repeat(3)
            The batch_size of interaction: 12
                k, torch.Size([12, 7]), cpu

        Returns:
            a copyed Interaction object with repeated Tensors.
        """
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].repeat(
                [sizes] + [1] * (len(self.interaction[k].shape) - 1)
            )
        return Interaction(ret)

    def repeat_interleave(self, repeats, dim=0):
        """Similar to repeat_interleave of PyTorch.

        Details can be found in:

            https://pytorch.org/docs/stable/tensors.html?highlight=repeat#torch.Tensor.repeat_interleave

        Note:
            ``torch.repeat_interleave()`` is supported in PyTorch >= 1.2.0.
        """
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].repeat_interleave(repeats, dim=dim)
        return Interaction(ret)

    def update(self, new_inter):
        """Similar to ``dict.update()``

        Args:
            new_inter (Interaction): current interaction will be updated by new_inter.
        """
        for k in new_inter.interaction:
            self.interaction[k] = new_inter.interaction[k]

    def drop(self, column):
        """Drop column in interaction.

        Args:
            column (str): the column to be dropped.
        """
        if column not in self.interaction:
            raise ValueError(f"Column [{column}] is not in [{self}].")
        del self.interaction[column]

    def _reindex(self, index):
        """Reset the index of interaction inplace.

        Args:
            index: the new index of current interaction.
        """
        for k in self.interaction:
            self.interaction[k] = self.interaction[k][index]

    def shuffle(self):
        """Shuffle current interaction inplace."""
        index = torch.randperm(self.length)
        self._reindex(index)

    def sort(self, by, ascending=True):
        """Sort the current interaction inplace.

        Args:
            by (str or list of str): Field that as the key in the sorting process.
            ascending (bool or list of bool, optional): Results are ascending if ``True``, otherwise descending.
                Defaults to ``True``
        """
        if isinstance(by, str):
            if by not in self.interaction:
                raise ValueError(f"[{by}] is not exist in interaction [{self}].")
            by = [by]
        elif isinstance(by, (list, tuple)):
            for b in by:
                if b not in self.interaction:
                    raise ValueError(f"[{b}] is not exist in interaction [{self}].")
        else:
            raise TypeError(f"Wrong type of by [{by}].")

        if isinstance(ascending, bool):
            ascending = [ascending]
        elif isinstance(ascending, (list, tuple)):
            for a in ascending:
                if not isinstance(a, bool):
                    raise TypeError(f"Wrong type of ascending [{ascending}].")
        else:
            raise TypeError(f"Wrong type of ascending [{ascending}].")

        if len(by) != len(ascending):
            if len(ascending) == 1:
                ascending = ascending * len(by)
            else:
                raise ValueError(
                    f"by [{by}] and ascending [{ascending}] should have same length."
                )

        for b, a in zip(by[::-1], ascending[::-1]):
            if len(self.interaction[b].shape) == 1:
                key = self.interaction[b]
            else:
                key = self.interaction[b][..., 0]
            index = np.argsort(key, kind="stable")
            if not a:
                index = torch.tensor(np.array(index)[::-1])
            self._reindex(index)

    def add_prefix(self, prefix):
        """Add prefix to current interaction's columns.

        Args:
            prefix (str): The prefix to be added.
        """
        self.interaction = {
            prefix + key: value for key, value in self.interaction.items()
        }


def cat_interactions(interactions):
    """Concatenate list of interactions to single interaction.

    Args:
        interactions (list of :class:`Interaction`): List of interactions to be concatenated.

    Returns:
        :class:`Interaction`: Concatenated interaction.
    """
    if not isinstance(interactions, (list, tuple)):
        raise TypeError(f"Interactions [{interactions}] should be list or tuple.")
    if len(interactions) == 0:
        raise ValueError(
            f"Interactions [{interactions}] should have some interactions."
        )

    columns_set = set(interactions[0].columns)
    for inter in interactions:
        if columns_set != set(inter.columns):
            raise ValueError(
                f"Interactions [{interactions}] should have some interactions."
            )

    new_inter = {
        col: torch.cat([inter[col] for inter in interactions]) for col in columns_set
    }
    return Interaction(new_inter)
