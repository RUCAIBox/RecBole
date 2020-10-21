# @Time   : 2020/7/10
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time    : 2020/9/15, 2020/9/16, 2020/8/12
# @Author  : Yupeng Hou, Yushuo Chen, Xingyu Pan
# @email   : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, panxy@ruc.edu.cn

"""
recbole.data.interaction
############################
"""

import numpy as np


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
        interaction (dict): keys are meaningful str (also can be called field name),
            and values are Torch Tensor of numpy Array with shape (batch_size, \\*).

        pos_len_list (list, optional): length of the list is the number of users in this batch,
            each value represents the number of a user's **positive** records. The order of the
            represented users should correspond to the order in the interaction.

        user_len_list (list, optional): length of the list is the number of users in this batch,
            each value represents the number of a user's **all** records. The order of the
            represented users should correspond to the order in the interaction.
    """

    def __init__(self, interaction, pos_len_list=None, user_len_list=None):
        self.interaction = interaction
        self.pos_len_list = pos_len_list
        self.user_len_list = user_len_list
        if (self.pos_len_list is None) ^ (self.user_len_list is None):
            raise ValueError('pos_len_list and user_len_list should be both None or valued.')
        for k in self.interaction:
            self.length = self.interaction[k].shape[0]
            break

    def __iter__(self):
        return self.interaction.__iter__()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.interaction[index]
        else:
            ret = {}
            for k in self.interaction:
                ret[k] = self.interaction[k][index]
            return Interaction(ret)

    def __len__(self):
        return self.length

    def __str__(self):
        info = ['The batch_size of interaction: {}'.format(self.length)]
        for k in self.interaction:
            temp_str = "    {}, {}, {}".format(k, self.interaction[k].shape, self.interaction[k].device.type)
            info.append(temp_str)
        info.append('\n')
        return '\n'.join(info)

    def __repr__(self):
        return self.__str__()

    def to(self, device, selected_field=None):
        """Transfer Tensors in this Interaction object to the specified device.

        Args:
            device (torch.device): target device.
            selected_field (str or iterable object, optional): if specified, only Tensors
            with keys in selected_field will be sent to device.

        Returns:
            Interaction: a copyed Interaction object with Tensors which are sented to
            the specified device.
        """
        ret = {}
        if isinstance(selected_field, str):
            selected_field = [selected_field]
        try:
            selected_field = set(selected_field)
            for k in self.interaction:
                if k in selected_field:
                    ret[k] = self.interaction[k].to(device)
                else:
                    ret[k] = self.interaction[k]
        except:
            for k in self.interaction:
                ret[k] = self.interaction[k].to(device)
        return Interaction(ret)

    def cpu(self):
        """Transfer Tensors in this Interaction object to cpu.

        Returns:
            Interaction: a copyed Interaction object with Tensors which are sented to cpu.
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
            if len(self.interaction[k].shape) == 1:
                ret[k] = self.interaction[k].repeat(sizes)
            else:
                ret[k] = self.interaction[k].repeat([sizes, 1])
        new_pos_len_list = self.pos_len_list * sizes if self.pos_len_list else None
        new_user_len_list = self.user_len_list * sizes if self.user_len_list else None
        return Interaction(ret, new_pos_len_list, new_user_len_list)

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
        new_pos_len_list = list(np.multiply(self.pos_len_list, repeats)) if self.pos_len_list else None
        new_user_len_list = list(np.multiply(self.user_len_list, repeats)) if self.user_len_list else None
        return Interaction(ret, new_pos_len_list, new_user_len_list)

    def update(self, new_inter):
        """Similar to ``dict.update()``
        """
        for k in new_inter.interaction:
            self.interaction[k] = new_inter.interaction[k]
        self.pos_len_list = new_inter.pos_len_list
        self.user_len_list = new_inter.user_len_list
