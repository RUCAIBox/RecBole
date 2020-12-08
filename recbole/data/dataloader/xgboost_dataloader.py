# @Time   : 2020/11/19
# @Author : Chen Yang
# @Email  : 254170321@qq.com

# UPDATE:
# @Time   : 2020/11/19
# @Author : Chen Yang
# @Email  : 254170321@qq.com

"""
recbole.data.dataloader.xgboost_dataloader
################################################
"""

from recbole.data.dataloader.general_dataloader import GeneralDataLoader, GeneralNegSampleDataLoader, GeneralFullDataLoader


class XgboostDataLoader(GeneralDataLoader):
    """:class:`XgboostDataLoader` is inherit from :class:`~recbole.data.dataloader.general_dataloader.GeneralDataLoader`,
    and didn't add/change anything at all.
    """
    pass


class XgboostNegSampleDataLoader(GeneralNegSampleDataLoader):
    """:class:`XgboostNegSampleDataLoader` is inherit from
    :class:`~recbole.data.dataloader.general_dataloader.GeneralNegSampleDataLoader`,
    and didn't add/change anything at all.
    """
    pass


class XgboostFullDataLoader(GeneralFullDataLoader):
    """:class:`XgboostFullDataLoader` is inherit from
    :class:`~recbole.data.dataloader.general_dataloader.GeneralFullDataLoader`,
    and didn't add/change anything at all.
    """
    pass
