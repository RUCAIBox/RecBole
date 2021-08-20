# @Time   : 2020/9/23
# @Author : Xingyu Pan
# @Email  : panxingyu@ruc.edu.cn

"""
recbole.data.kg_seq_dataset
#############################
"""

from recbole.data.dataset import SequentialDataset, KnowledgeBasedDataset


class KGSeqDataset(SequentialDataset, KnowledgeBasedDataset):
    """Containing both processing of Sequential Models and Knowledge-based Models.

    Inherit from :class:`~recbole.data.dataset.sequential_dataset.SequentialDataset` and
    :class:`~recbole.data.dataset.kg_dataset.KnowledgeBasedDataset`.
    """

    def __init__(self, config):
        super().__init__(config)
