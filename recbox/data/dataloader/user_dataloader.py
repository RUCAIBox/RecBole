# @Time   : 2020/9/23
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE
# @Time   : 2020/9/23
# @Author : Yushuo Chen
# @email  : chenyushuo@ruc.edu.cn


from recbox.data.dataloader import AbstractDataLoader
from recbox.utils.enum_type import DataLoaderType, InputType


class UserDataLoader(AbstractDataLoader):
    dl_type = DataLoaderType.ORIGIN

    def __init__(self, config, dataset,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        self.uid_field = dataset.uid_field

        super().__init__(config=config, dataset=dataset,
                         batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

    def setup(self):
        if self.shuffle is False:
            self.shuffle = True
            self.logger.warning('UserDataLoader must shuffle the data')

    @property
    def pr_end(self):
        return len(self.dataset.user_feat)

    def _shuffle(self):
        self.dataset.user_feat = self.dataset.user_feat.sample(frac=1).reset_index(drop=True)

    def _next_batch_data(self):
        cur_data = self.dataset.user_feat[[self.uid_field]][self.pr: self.pr + self.step]
        self.pr += self.step
        return self._dataframe_to_interaction(cur_data)
