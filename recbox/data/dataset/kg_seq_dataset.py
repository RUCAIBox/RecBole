# @Time   : 2020/9/23
# @Author : Xingyu Pan
# @Email  : panxingyu@ruc.edu.cn

from recbox.data.dataset import SequentialDataset, KnowledgeBasedDataset
from recbox.utils import FeatureSource, FeatureType


class Kg_Seq_Dataset(SequentialDataset, KnowledgeBasedDataset):
      def __init__(self, config, saved_dataset=None):
         super().__init__(config, saved_dataset=saved_dataset) 
      