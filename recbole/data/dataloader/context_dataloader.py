# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2020/9/9, 2020/9/16
# @Author : Yupeng Hou, Yushuo Chen
# @email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn

"""
recbole.data.dataloader.context_dataloader
################################################
"""

from recbole.data.dataloader.general_dataloader import GeneralDataLoader, GeneralNegSampleDataLoader


class ContextDataLoader(GeneralDataLoader):
    """:class:`ContextDataLoader` is inherit from :class:`~recbole.data.dataloader.general_dataloader.GeneralDataLoader`,
    and didn't add/change anything at all.
    """
    pass


class ContextNegSampleDataLoader(GeneralNegSampleDataLoader):
    """:class:`ContextNegSampleDataLoader` is inherit from
    :class:`~recbole.data.dataloader.general_dataloader.GeneralNegSampleDataLoader`,
    and didn't add/change anything at all.
    """
    pass
