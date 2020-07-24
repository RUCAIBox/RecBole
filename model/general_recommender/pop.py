import numpy as np
import torch
from torch.autograd import Variable
from collections import defaultdict
from model.abstract_recommender import GeneralRecommender


class Pop(GeneralRecommender):
    def __init__(self, config, dataset):
        super(Pop, self).__init__()
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.device = config['device']

        self.item_cnt = defaultdict(int)
        self.max_cnt = 0

        self.fake_loss = torch.nn.Parameter(torch.FloatTensor([2]))

    def calculate_loss(self, interaction):

        item = interaction[self.ITEM_ID]
        item = item.cpu().numpy()
        for i in item:
            self.item_cnt[int(i)] += 1
            self.max_cnt = max(self.max_cnt, self.item_cnt[(int(i))])

        return self.fake_loss

    def predict(self, interaction):

        item = interaction[self.ITEM_ID]
        item = item.cpu().numpy()
        result_score = []
        for i in item:
            i = int(i)
            if i not in self.item_cnt:
                # todo: how to deal with item that not be seen in train_data
                # raise RuntimeError('New item can not get popularity score!')
                score = 0.0
            else:
                score = self.item_cnt[i]/self.max_cnt
            result_score.append(score)
        result = torch.from_numpy(np.array(result_score)).to(self.device)

        return result
