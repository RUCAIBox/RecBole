# @Time   : 2020/9/23
# @Author : Xingyu Pan
# @Email  : panxingyu@ruc.edu.cn

"""
recbole.data.kg_seq_dataset
#############################
"""

from recbole.data.dataset import SequentialDataset, KnowledgeBasedDataset


class Kg_Seq_Dataset(SequentialDataset, KnowledgeBasedDataset):
    """Containing both processing of Sequential Models and Knowledge-based Models.

    Inherit from :class:`~recbole.data.dataset.sequential_dataset.SequentialDataset` and
    :class:`~recbole.data.dataset.kg_dataset.KnowledgeBasedDataset`.
    """
    def __init__(self, config, saved_dataset=None):
        super().__init__(config, saved_dataset=saved_dataset)
