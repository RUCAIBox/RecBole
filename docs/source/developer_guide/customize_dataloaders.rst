Customize DataLoaders
======================
Here, we present how to develop a new DataLoader, and apply it into our tool. If we have a new model,
and there is no special requirement for loading the data, then we need to design a new DataLoader.


Abstract DataLoader
--------------------------
In this project, there are three abstracts: :class:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader`,
:class:`~recbole.data.dataloader.neg_sample_mixin.NegSampleMixin`, :class:`~recbole.data.dataloader.neg_sample_mixin.NegSampleByMixin`.

In general, the new dataloader should inherit from the above three abstract classes.
If one only needs to modify existing DataLoader, you can also inherit from the it.
The documentation of dataloader: :doc:`../../recbole/recbole.data.dataloader`


AbstractDataLoader
^^^^^^^^^^^^^^^^^^^^^^^^^^
:class:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader` is the most basic abstract class,
which includes three functions: :meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.pr_end`,
:meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader._shuffle`
and :meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader._next_batch_data`.
:meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.pr_end` is the max
:attr:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.pr` plus 1.
:meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader._shuffle` is leverage to permute the dataset,
which will be invoked by :meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.__iter__`
if the parameter :attr:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.shuffle` is True.
:meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader._next_batch_data` is used to
load the next batch data, and return the :class:`~recbole.data.interaction.Interaction` format,
which will be invoked in :meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.__next__`.

In :class:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader`,
there are two functions to assist the conversion of :meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader._next_batch_data`,
one is :meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader._dataframe_to_interaction`,
and the other is :meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader._dict_to_interaction`.
They both use the functions with the same name in :attr:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.dataset`.
The :class:`pandas.DataFrame` or :class:`dict` is converted into :class:`~recbole.data.interaction.Interaction`.

In addition to the above three functions, two other functions can also be rewrite,
that is :meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.setup`
and :meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.data_preprocess`.

:meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.setup` is used to tackle the problems except initializing the parameters.
For example, reset the :attr:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.batch_size`,
examine the :attr:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.shuffle` setting.
All these things can be rewritten in the subclass.
:meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.data_preprocess` is used to process the data,
e.g., negative sampling.

At the end of :meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.__init__`,
:meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.setup` will be invoked,
and then if :attr:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.real_time` is ``True``,
then :meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.data_preprocess` is recalled.

NegSampleMixin
^^^^^^^^^^^^^^^^^^^^^^^^^^
:class:`~recbole.data.dataloader.neg_sample_mixin.NegSampleMixin` inherent from
:class:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader`, which is used for negative sampling.
It has three additional functions upon its father class:
:meth:`~recbole.data.dataloader.neg_sample_mixin.NegSampleMixin._batch_size_adaptation`,
:meth:`~recbole.data.dataloader.neg_sample_mixin.NegSampleMixin._neg_sampling`
and :meth:`~recbole.data.dataloader.neg_sample_mixin.NegSampleMixin.get_pos_len_list`.

Since the positive and negative samples should be framed in the same batch,
the original batch size can be not appropriate.
:meth:`~recbole.data.dataloader.neg_sample_mixin.NegSampleMixin._batch_size_adaptation` is used to reset the batch size,
such that the positive and negative samples can be in the same batch.
:meth:`~recbole.data.dataloader.neg_sample_mixin.NegSampleMixin._neg_sampling` is used for negative sampling,
which should be implemented by the subclass.
:meth:`~recbole.data.dataloader.neg_sample_mixin.NegSampleMixin.get_pos_len_list` returns the positive sample number for each user.

In addition, :meth:`~recbole.data.dataloader.neg_sample_mixin.NegSampleMixin.setup`
and :meth:`~recbole.data.dataloader.neg_sample_mixin.NegSampleMixin.data_preprocess` are also changed.
:meth:`~recbole.data.dataloader.neg_sample_mixin.NegSampleMixin.setup` will
call :meth:`~recbole.data.dataloader.neg_sample_mixin.NegSampleMixin._batch_size_adaptation`,
:meth:`~recbole.data.dataloader.neg_sample_mixin.NegSampleMixin.data_preprocess` is used for negative sampling
which should be implemented in the subclass.

NegSampleByMixin
^^^^^^^^^^^^^^^^^^^^^^^^^^
:class:`~recbole.data.dataloader.neg_sample_mixin.NegSampleByMixin` inherent
from :class:`~recbole.data.dataloader.neg_sample_mixin.NegSampleMixin`,
which is used for negative sampling by ratio.
It supports two strategies, the first one is ``pair-wise sampling``, the other is ``point-wise sampling``.
Then based on the parent class, two functions are added:
:meth:`~recbole.data.dataloader.neg_sample_mixin.NegSampleByMixin._neg_sample_by_pair_wise_sampling`
and :meth:`~recbole.data.dataloader.neg_sample_mixin.NegSampleByMixin._neg_sample_by_point_wise_sampling`.


Example
--------------------------
Here, we take :class:`~recbole.data.dataloader.user_dataloader.UserDataLoader` as the example,
this dataloader returns user id, which is leveraged to train the user representations.


Implement __init__()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:meth:`__init__` can be used to initialize some of the necessary parameters.
Here, we just need to record :attr:`uid_field`.

.. code:: python

    def __init__(self, config, dataset,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        self.uid_field = dataset.uid_field

        super().__init__(config=config, dataset=dataset,
                         batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

Implement setup()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Because of some training requirement, :attr:`self.shuffle` should be true.
Then we can check and revise :attr:`self.shuffle` in :meth:`~recbole.data.dataloader.user_dataloader.setup`.


.. code:: python

    def setup(self):
        if self.shuffle is False:
            self.shuffle = True
            self.logger.warning('UserDataLoader must shuffle the data')

Implement pr_end() and _shuffle()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Since this dataloader only returns user id, these function can be implemented readily.

.. code:: python

    @property
    def pr_end(self):
        return len(self.dataset.user_feat)

    def _shuffle(self):
        self.dataset.user_feat = self.dataset.user_feat.sample(frac=1).reset_index(drop=True)

Implement _next_batch_data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This function only require return user id from :attr:`user_feat`,
we only have to select one column, and use :meth:`_dataframe_to_interaction` to convert
:class:`pandas.DataFrame` into :class:`~recbole.data.interaction.Interaction`.


.. code:: python

    def _next_batch_data(self):
        cur_data = self.dataset.user_feat[[self.uid_field]][self.pr: self.pr + self.step]
        self.pr += self.step
        return self._dataframe_to_interaction(cur_data)


Complete Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    class UserDataLoader(AbstractDataLoader):
        """:class:`UserDataLoader` will return a batch of data which only contains user-id when it is iterated.

        Args:
            config (Config): The config of dataloader.
            dataset (Dataset): The dataset of dataloader.
            batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
            dl_format (InputType, optional): The input type of dataloader. Defaults to
                :obj:`~recbole.utils.enum_type.InputType.POINTWISE`.
            shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

        Attributes:
            shuffle (bool): Whether the dataloader will be shuffle after a round.
                However, in :class:`UserDataLoader`, it's guaranteed to be ``True``.
        """
        dl_type = DataLoaderType.ORIGIN

        def __init__(self, config, dataset,
                     batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
            self.uid_field = dataset.uid_field

            super().__init__(config=config, dataset=dataset,
                             batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

        def setup(self):
            """Make sure that the :attr:`shuffle` is True. If :attr:`shuffle` is False, it will be changed to True
            and give a warning to user.
            """
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


Other more complex Dataloader development can refer to the source code.
