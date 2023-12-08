# -*- coding: utf-8 -*-
# @Time   : 2022/7/19
# @Author : Gaowei Zhang
# @Email  : zgw15630559577@163.com

# UPDATE
# @Time    : 2023/12/8
# @Author  : Bingqian Li
# @email   : lizibing666@gmail.com

import math
import numpy as np
import random
import torch
from copy import deepcopy
from recbole.data.interaction import Interaction, cat_interactions


def construct_transform(config):
    """
    Transformation for batch data.
    """
    if config["transform"] is None:
        return Equal(config)
    else:
        str2transform = {
            "mask_itemseq": MaskItemSequence,
            "inverse_itemseq": InverseItemSequence,
            "crop_itemseq": CropItemSequence,
            "reorder_itemseq": ReorderItemSequence,
            "user_defined": UserDefinedTransform,
            "random_itemseq": RandomAugmentationSequence
        }
        if config["transform"] not in str2transform:
            raise NotImplementedError(
                f"There is no transform named '{config['transform']}'"
            )

        return str2transform[config["transform"]](config)


class Equal:
    def __init__(self, config):
        pass

    def __call__(self, dataset, interaction):
        return interaction


class MaskItemSequence:
    """
    Mask item sequence for training.
    """

    def __init__(self, config):
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.MASK_ITEM_SEQ = "Mask_" + self.ITEM_SEQ
        self.POS_ITEMS = "Pos_" + config["ITEM_ID_FIELD"]
        self.NEG_ITEMS = "Neg_" + config["ITEM_ID_FIELD"]
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.mask_ratio = config["mask_ratio"]
        self.ft_ratio = 0 if not hasattr(config, "ft_ratio") else config["ft_ratio"]
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)
        self.MASK_INDEX = "MASK_INDEX"
        config["MASK_INDEX"] = "MASK_INDEX"
        config["MASK_ITEM_SEQ"] = self.MASK_ITEM_SEQ
        config["POS_ITEMS"] = self.POS_ITEMS
        config["NEG_ITEMS"] = self.NEG_ITEMS
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.config = config

    def _neg_sample(self, item_set, n_items):
        item = random.randint(1, n_items - 1)
        while item in item_set:
            item = random.randint(1, n_items - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence

    def _append_mask_last(self, interaction, n_items, device):
        batch_size = interaction[self.ITEM_SEQ].size(0)
        pos_items, neg_items, masked_index, masked_item_sequence = [], [], [], []
        seq_instance = interaction[self.ITEM_SEQ].cpu().numpy().tolist()
        item_seq_len = interaction[self.ITEM_SEQ_LEN].cpu().numpy().tolist()
        for instance, lens in zip(seq_instance, item_seq_len):
            mask_seq = instance.copy()
            ext = instance[lens - 1]
            mask_seq[lens - 1] = n_items
            masked_item_sequence.append(mask_seq)
            pos_items.append(self._padding_sequence([ext], self.mask_item_length))
            neg_items.append(
                self._padding_sequence(
                    [self._neg_sample(instance, n_items)], self.mask_item_length
                )
            )
            masked_index.append(
                self._padding_sequence([lens - 1], self.mask_item_length)
            )
        # [B Len]
        masked_item_sequence = torch.tensor(
            masked_item_sequence, dtype=torch.long, device=device
        ).view(batch_size, -1)
        # [B mask_len]
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(
            batch_size, -1
        )
        # [B mask_len]
        neg_items = torch.tensor(neg_items, dtype=torch.long, device=device).view(
            batch_size, -1
        )
        # [B mask_len]
        masked_index = torch.tensor(masked_index, dtype=torch.long, device=device).view(
            batch_size, -1
        )
        new_dict = {
            self.MASK_ITEM_SEQ: masked_item_sequence,
            self.POS_ITEMS: pos_items,
            self.NEG_ITEMS: neg_items,
            self.MASK_INDEX: masked_index,
        }
        ft_interaction = deepcopy(interaction)
        ft_interaction.update(Interaction(new_dict))
        return ft_interaction

    def __call__(self, dataset, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        device = item_seq.device
        batch_size = item_seq.size(0)
        n_items = dataset.num(self.ITEM_ID)
        sequence_instances = item_seq.cpu().numpy().tolist()

        # Masked Item Prediction
        # [B * Len]
        masked_item_sequence = []
        pos_items = []
        neg_items = []
        masked_index = []

        if random.random() < self.ft_ratio:
            interaction = self._append_mask_last(interaction, n_items, device)
        else:
            for instance in sequence_instances:
                # WE MUST USE 'copy()' HERE!
                masked_sequence = instance.copy()
                pos_item = []
                neg_item = []
                index_ids = []
                for index_id, item in enumerate(instance):
                    # padding is 0, the sequence is end
                    if item == 0:
                        break
                    prob = random.random()
                    if prob < self.mask_ratio:
                        pos_item.append(item)
                        neg_item.append(self._neg_sample(instance, n_items))
                        masked_sequence[index_id] = n_items
                        index_ids.append(index_id)

                masked_item_sequence.append(masked_sequence)
                pos_items.append(
                    self._padding_sequence(pos_item, self.mask_item_length)
                )
                neg_items.append(
                    self._padding_sequence(neg_item, self.mask_item_length)
                )
                masked_index.append(
                    self._padding_sequence(index_ids, self.mask_item_length)
                )

            # [B Len]
            masked_item_sequence = torch.tensor(
                masked_item_sequence, dtype=torch.long, device=device
            ).view(batch_size, -1)
            # [B mask_len]
            pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(
                batch_size, -1
            )
            # [B mask_len]
            neg_items = torch.tensor(neg_items, dtype=torch.long, device=device).view(
                batch_size, -1
            )
            # [B mask_len]
            masked_index = torch.tensor(
                masked_index, dtype=torch.long, device=device
            ).view(batch_size, -1)
            new_dict = {
                self.MASK_ITEM_SEQ: masked_item_sequence,
                self.POS_ITEMS: pos_items,
                self.NEG_ITEMS: neg_items,
                self.MASK_INDEX: masked_index,
            }
            interaction.update(Interaction(new_dict))
        return interaction


class InverseItemSequence:
    """
    inverse the seq_item, like this
        [1,2,3,0,0,0,0] -- after inverse -->> [0,0,0,0,1,2,3]
    """

    def __init__(self, config):
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.INVERSE_ITEM_SEQ = "Inverse_" + self.ITEM_SEQ
        config["INVERSE_ITEM_SEQ"] = self.INVERSE_ITEM_SEQ

    def __call__(self, dataset, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        device = item_seq.device
        item_seq = item_seq.cpu().numpy()
        item_seq_len = item_seq_len.cpu().numpy()
        new_item_seq = []
        for items, length in zip(item_seq, item_seq_len):
            item = list(items[:length])
            zeros = list(items[length:])
            seqs = zeros + item
            new_item_seq.append(seqs)
        inverse_item_seq = torch.tensor(new_item_seq, dtype=torch.long, device=device)
        new_dict = {self.INVERSE_ITEM_SEQ: inverse_item_seq}
        interaction.update(Interaction(new_dict))
        return interaction

class RandomAugmentationSequence:
    def __init__(self, config):
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.RANDOM_ITEM_SEQ = "Random_" + self.ITEM_SEQ
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]        
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        config["RANDOM_ITEM_SEQ"] = self.RANDOM_ITEM_SEQ

    
    def __call__(self, dataset, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        device = item_seq.device
        n_items = dataset.num(self.ITEM_ID)

        aug_seq1 = []
        aug_len1 = []
        aug_seq2 = []
        aug_len2 = []

        for seq, length in zip(item_seq, item_seq_len):
            if length > 1:
                switch = random.sample(range(3), k=2)
            else:
                switch = [3, 3]
                aug_seq = seq
                aug_len = length
            if switch[0] == 0:
                aug_seq, aug_len = self.item_crop(seq, length)
            elif switch[0] == 1:
                aug_seq, aug_len = self.item_mask(seq, n_items, length)
            elif switch[0] == 2:
                aug_seq, aug_len = self.item_reorder(seq, length)
            
            aug_seq1.append(aug_seq)
            aug_len1.append(aug_len)
            
            if switch[1] == 0:
                aug_seq, aug_len = self.item_crop(seq, length)
            elif switch[1] == 1:
                aug_seq, aug_len = self.item_mask(seq, n_items, length)
            elif switch[1] == 2:
                aug_seq, aug_len = self.item_reorder(seq, length)

            aug_seq2.append(aug_seq)
            aug_len2.append(aug_len)
        
        new_dict = {
            "aug1" : torch.stack(aug_seq1),
            "aug1_len" : torch.stack(aug_len1),
            "aug2" : torch.stack(aug_seq2),
            "aug2_len" : torch.stack(aug_len2)
        }
        interaction.update(Interaction(new_dict))
        return interaction
        
    def item_crop(self, item_seq, item_seq_len, eta=0.6):
        num_left = math.floor(item_seq_len * eta)
        crop_begin = random.randint(0, item_seq_len - num_left)
        croped_item_seq = np.zeros(item_seq.shape[0])
        if crop_begin + num_left < item_seq.shape[0]:
            croped_item_seq[:num_left] = item_seq.cpu().detach().numpy()[crop_begin:crop_begin + num_left]
        else:
            croped_item_seq[:num_left] = item_seq.cpu().detach().numpy()[crop_begin:]
        return torch.tensor(croped_item_seq, dtype=torch.long, device=item_seq.device),\
               torch.tensor(num_left, dtype=torch.long, device=item_seq.device)

    def item_mask(self, item_seq, n_items, item_seq_len, gamma=0.3):
        num_mask = math.floor(item_seq_len * gamma)
        mask_index = random.sample(range(item_seq_len), k=num_mask)
        masked_item_seq = item_seq.cpu().detach().numpy().copy()
        masked_item_seq[mask_index] = n_items - 1 # token 0 has been used for semantic masking
        return torch.tensor(masked_item_seq, dtype=torch.long, device=item_seq.device), item_seq_len

    def item_reorder(self, item_seq, item_seq_len, beta=0.6):
        num_reorder = math.floor(item_seq_len * beta)
        reorder_begin = random.randint(0, item_seq_len - num_reorder)
        reordered_item_seq = item_seq.cpu().detach().numpy().copy()
        shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
        random.shuffle(shuffle_index)
        reordered_item_seq[reorder_begin:reorder_begin + num_reorder] = reordered_item_seq[shuffle_index]
        return torch.tensor(reordered_item_seq, dtype=torch.long, device=item_seq.device), item_seq_len


class CropItemSequence:
    """
    Random crop for item sequence.
    """

    def __init__(self, config):
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.CROP_ITEM_SEQ = "Crop_" + self.ITEM_SEQ
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.CROP_ITEM_SEQ_LEN = self.CROP_ITEM_SEQ + self.ITEM_SEQ_LEN
        self.crop_eta = config["eta"]
        config["CROP_ITEM_SEQ"] = self.CROP_ITEM_SEQ
        config["CROP_ITEM_SEQ_LEN"] = self.CROP_ITEM_SEQ_LEN

    def __call__(self, dataset, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        device = item_seq.device
        crop_item_seq_list, crop_item_seqlen_list = [], []

        for seq, length in zip(item_seq, item_seq_len):
            crop_len = math.floor(length * self.crop_eta)
            crop_begin = random.randint(0, length - crop_len)
            crop_item_seq = np.zeros(seq.shape[0])
            if crop_begin + crop_len < seq.shape[0]:
                crop_item_seq[:crop_len] = seq[crop_begin : crop_begin + crop_len]
            else:
                crop_item_seq[:crop_len] = seq[crop_begin:]
            crop_item_seq_list.append(
                torch.tensor(crop_item_seq, dtype=torch.long, device=device)
            )
            crop_item_seqlen_list.append(
                torch.tensor(crop_len, dtype=torch.long, device=device)
            )
        new_dict = {
            self.CROP_ITEM_SEQ: torch.stack(crop_item_seq_list),
            self.CROP_ITEM_SEQ_LEN: torch.stack(crop_item_seqlen_list),
        }
        interaction.update(Interaction(new_dict))
        return interaction


class ReorderItemSequence:
    """
    Reorder operation for item sequence.
    """

    def __init__(self, config):
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.REORDER_ITEM_SEQ = "Reorder_" + self.ITEM_SEQ
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.reorder_beta = config["beta"]
        config["REORDER_ITEM_SEQ"] = self.REORDER_ITEM_SEQ

    def __call__(self, dataset, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        device = item_seq.device
        reorder_seq_list = []

        for seq, length in zip(item_seq, item_seq_len):
            reorder_len = math.floor(length * self.reorder_beta)
            reorder_begin = random.randint(0, length - reorder_len)
            reorder_item_seq = seq.cpu().detach().numpy().copy()

            shuffle_index = list(range(reorder_begin, reorder_begin + reorder_len))
            random.shuffle(shuffle_index)
            reorder_item_seq[
                reorder_begin : reorder_begin + reorder_len
            ] = reorder_item_seq[shuffle_index]

            reorder_seq_list.append(
                torch.tensor(reorder_item_seq, dtype=torch.long, device=device)
            )
        new_dict = {self.REORDER_ITEM_SEQ: torch.stack(reorder_seq_list)}
        interaction.update(Interaction(new_dict))
        return interaction


class UserDefinedTransform:
    def __init__(self, config):
        pass

    def __call__(self, dataset, interaction):
        pass
