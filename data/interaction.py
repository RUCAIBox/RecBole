# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : interaction.py

class Interaction(object):
    def __init__(self, interaction):
        self.interaction = interaction
        for k in self.interaction:
            self.length = self.interaction[k].shape[0]
            break

    def __getitem__(self, index):
        return self.interaction[index]

    def __len__(self):
        return self.length

    def to(self, device):
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].to(device)
        return Interaction(ret)
