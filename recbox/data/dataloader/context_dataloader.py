# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2020/9/9, 2020/9/12
# @Author : Yupeng Hou, Yushuo Chen
# @email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn


from .general_dataloader import GeneralDataLoader, GeneralNegSampleDataLoader


class ContextDataLoader(GeneralDataLoader):
    pass


class ContextNegSampleDataLoader(GeneralNegSampleDataLoader):
    pass
