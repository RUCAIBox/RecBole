# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2020/9/18, 2020/9/21, 2020/8/31
# @Author : Yupeng Hou, Yushuo Chen, Kaiyuan Li
# @email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, tsotfsk@outlook.com

"""
recbole.data.dataloader.knowledge_dataloader
################################################
"""

from recbole.data.dataloader import AbstractDataLoader, GeneralNegSampleDataLoader
from recbole.data.interaction import Interaction
from recbole.utils import InputType, KGDataLoaderState


class KGDataLoader(AbstractDataLoader):
    """:class:`KGDataLoader` is a dataloader which would return the triplets with negative examples
    in a knowledge graph.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (KGSampler): The knowledge graph sampler of dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~recbole.utils.InputType.PAIRWISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

    Attributes:
        shuffle (bool): Whether the dataloader will be shuffle after a round.
            However, in :class:`KGDataLoader`, it's guaranteed to be ``True``.
    """

    def __init__(self, config, dataset, sampler, batch_size=1, dl_format=InputType.PAIRWISE, shuffle=False):
        self.sampler = sampler
        self.neg_sample_num = 1

        self.neg_prefix = config['NEG_PREFIX']
        self.hid_field = dataset.head_entity_field
        self.tid_field = dataset.tail_entity_field

        # kg negative cols
        self.neg_tid_field = self.neg_prefix + self.tid_field
        dataset.copy_field_property(self.neg_tid_field, self.tid_field)

        super().__init__(config, dataset, batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

    def setup(self):
        """Make sure that the :attr:`shuffle` is True. If :attr:`shuffle` is False, it will be changed to True
        and give a warning to user.
        """
        if self.shuffle is False:
            self.shuffle = True
            self.logger.warning('kg based dataloader must shuffle the data')

    @property
    def pr_end(self):
        return len(self.dataset.kg_feat)

    def _shuffle(self):
        self.dataset.kg_feat.shuffle()

    def _next_batch_data(self):
        cur_data = self._neg_sampling(self.dataset.kg_feat[self.pr:self.pr + self.step])
        self.pr += self.step
        return cur_data

    def _neg_sampling(self, kg_feat):
        hids = kg_feat[self.hid_field]
        neg_tids = self.sampler.sample_by_entity_ids(hids, self.neg_sample_num)
        kg_feat.update(Interaction({self.neg_tid_field: neg_tids}))
        return kg_feat


class KnowledgeBasedDataLoader(AbstractDataLoader):
    """:class:`KnowledgeBasedDataLoader` is used for knowledge based model.

    It has three states, which is saved in :attr:`state`.
    In different states, :meth:`~_next_batch_data` will return different :class:`~recbole.data.interaction.Interaction`.
    Detailed, please see :attr:`~state`.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        kg_sampler (KGSampler): The knowledge graph sampler of dataloader.
        neg_sample_args (dict): The neg_sample_args of dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~recbole.utils.enum_type.InputType.POINTWISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

    Attributes:
        state (KGDataLoaderState): 
            This dataloader has three states:

                - :obj:`~recbole.utils.enum_type.KGDataLoaderState.RS`
                - :obj:`~recbole.utils.enum_type.KGDataLoaderState.KG`
                - :obj:`~recbole.utils.enum_type.KGDataLoaderState.RSKG`

            In the first state, this dataloader would only return the triplets with negative examples in a knowledge graph.

            In the second state, this dataloader would only return the user-item interaction.

            In the last state, this dataloader would return both knowledge graph information
            and user-item interaction information.
    """

    def __init__(
        self,
        config,
        dataset,
        sampler,
        kg_sampler,
        neg_sample_args,
        batch_size=1,
        dl_format=InputType.POINTWISE,
        shuffle=False
    ):

        # using sampler
        self.general_dataloader = GeneralNegSampleDataLoader(
            config=config,
            dataset=dataset,
            sampler=sampler,
            neg_sample_args=neg_sample_args,
            batch_size=batch_size,
            dl_format=dl_format,
            shuffle=shuffle
        )

        # using kg_sampler and dl_format is pairwise
        self.kg_dataloader = KGDataLoader(
            config, dataset, kg_sampler, batch_size=batch_size, dl_format=InputType.PAIRWISE, shuffle=True
        )

        self.state = None

        super().__init__(config, dataset, batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

    def __iter__(self):
        if self.state is None:
            raise ValueError(
                'The dataloader\'s state must be set when using the kg based dataloader, '
                'you should call set_mode() before __iter__()'
            )
        if self.state == KGDataLoaderState.KG:
            return self.kg_dataloader.__iter__()
        elif self.state == KGDataLoaderState.RS:
            return self.general_dataloader.__iter__()
        elif self.state == KGDataLoaderState.RSKG:
            self.kg_dataloader.__iter__()
            self.general_dataloader.__iter__()
            return self

    def _shuffle(self):
        pass

    def __next__(self):
        if self.general_dataloader.pr >= self.general_dataloader.pr_end:
            self.general_dataloader.pr = 0
            self.kg_dataloader.pr = 0
            raise StopIteration()
        return self._next_batch_data()

    def __len__(self):
        if self.state == KGDataLoaderState.KG:
            return len(self.kg_dataloader)
        else:
            return len(self.general_dataloader)

    @property
    def pr_end(self):
        if self.state == KGDataLoaderState.KG:
            return self.kg_dataloader.pr_end
        else:
            return self.general_dataloader.pr_end

    def _next_batch_data(self):
        try:
            kg_data = self.kg_dataloader.__next__()
        except StopIteration:
            kg_data = self.kg_dataloader.__next__()
        rec_data = self.general_dataloader.__next__()
        rec_data.update(kg_data)
        return rec_data

    def set_mode(self, state):
        """Set the mode of :class:`KnowledgeBasedDataLoader`, it can be set to three states:

            - KGDataLoaderState.RS
            - KGDataLoaderState.KG
            - KGDataLoaderState.RSKG

        The state of :class:`KnowledgeBasedDataLoader` would affect the result of _next_batch_data().

        Args:
            state (KGDataLoaderState): the state of :class:`KnowledgeBasedDataLoader`.
        """
        if state not in set(KGDataLoaderState):
            raise NotImplementedError(f'Kg data loader has no state named [{self.state}].')
        self.state = state
