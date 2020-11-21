# @Time   : 2020/11/19
# @Author : Chen Yang
# @Email  : 254170321@qq.com

# UPDATE:
# @Time   : 2020/11/19
# @Author : Chen Yang
# @Email  : 254170321@qq.com

"""
recbole.data.xgboost_dataloader
#############################
"""

from recbole.data.dataloader import AbstractDataLoader
from recbole.utils import InputType

class XgboostDataLoader(AbstractDataLoader):
    """:class:`XgboostDataLoader` is used for xgboost model and it return data in DMatrix form.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~recbole.utils.enum_type.InputType.POINTWISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

    Attribute:
        dataset (dataset): The dataset of this dataloader.
        dataset_DMatrix (DMatrix): The dataset in DMatrix form.

    """

    def __init__(self, config, dataset,
                batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        super().__init__(config, dataset,
                        batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)
        try:
            import xgboost as xgb
        except ImportError:
            self.logger.error('xgboost needs to be installed in order to use this module')
        self.dataset = dataset

        self.label_field = config['xgb_label_field']
        self.weight = config['xgb_weight']
        self.base_margin = config['xgb_base_margin']
        self.missing = config['xgb_missing']
        self.silent = config['xgb_silent']
        self.feature_names = config['xgb_feature_names']
        self.feature_types = config['xgb_feature_types']
        self.nthread = config['xgb_nthread']

        self.dataset_DMatrix = xgb.DMatrix(data = dataset.inter_feat.drop(self.label_field,axis=1,inplace=False), 
                                label = dataset.inter_feat[self.label_field], 
                                weight = self.weight,
                                base_margin = self.base_margin,
                                missing = self.missing, 
                                silent = self.silent, 
                                feature_names = self.feature_names, 
                                feature_types = self.feature_types, 
                                nthread = self.nthread)

    
    


