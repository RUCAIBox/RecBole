Customize DataLoaders
======================
Here, we present how to develop a new DataLoader, and apply it into our tool. If we have a new model,
and there is special requirement for loading the data, then we need to design a new DataLoader.


Abstract DataLoader
--------------------------
In this project, there are two abstract dataloaders:
:class:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader`,
:class:`~recbole.data.dataloader.abstract_dataloader.NegSampleDataLoader`.

In general, the new dataloader should inherit from the above two abstract classes.
If one only needs to modify existing DataLoader, you can also inherit from it.
The documentation of dataloader: :doc:`../../recbole/recbole.data.dataloader`


AbstractDataLoader
^^^^^^^^^^^^^^^^^^^^^^^^^^
:class:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader` is the most basic abstract class,
which includes three important attributes:
:attr:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.pr`,
:attr:`~recbole.data.dataloader.abstract_dataloader.AbstractDataloader.batch_size` and
:attr:`~recbole.data.dataloader.abstract_dataloader.AbstractDataloader.step`.
The :attr:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.pr`
represents the pointer of this dataloader.
The :attr:`~recbole.data.dataloader.abstract_dataloader.AbstractDataloader.batch_size`
represents the upper bound of the number of interactions in one single batch.
And the :attr:`~recbole.data.dataloader.abstract_dataloader.AbstractDataloader.step`
represents the increment of :attr:`~recbole.data.dataloader.abstract_dataloader.AbstractDataloader.pr` for each batch.

And :class:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader` includes four functions to be implemented:
:meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader._init_batch_size_and_step`,
:meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.pr_end`,
:meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader._shuffle`
and :meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader._next_batch_data`.
:meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader._init_batch_size_and_step` is used to
initialize :attr:`~recbole.data.dataloader.abstract_dataloader.AbstractDataloader.batch_size` and
:attr:`~recbole.data.dataloader.abstract_dataloader.AbstractDataloader.step`.
:meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.pr_end` is the max
:attr:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.pr` plus 1.
:meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader._shuffle` is leveraged to permute the dataset,
which will be invoked by :meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.__iter__`
if the parameter :attr:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.shuffle` is True.
:meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader._next_batch_data` is used to
load the next batch data, and return the :class:`~recbole.data.interaction.Interaction` format,
which will be invoked in :meth:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader.__next__`.


NegSampleDataLoader
^^^^^^^^^^^^^^^^^^^^^^^^^^
:class:`~recbole.data.dataloader.abstract_dataloader.NegSampleDataLoader` inherents from
:class:`~recbole.data.dataloader.abstract_dataloader.AbstractDataLoader`, which is used for negative sampling.
It has four additional functions upon its parent class:
:meth:`~recbole.data.dataloader.abstract_dataloader.NegSampleDataLoader._set_neg_sample_args`,
:meth:`~recbole.data.dataloader.abstract_dataloader.NegSampleDataLoader._neg_sampling`,
:meth:`~recbole.data.dataloader.abstract_dataloader.NegSampleDataLoader._neg_sample_by_pair_wise_sampling`,
and :meth:`~recbole.data.dataloader.abstract_dataloader.NegSampleDataLoader._neg_sample_by_point_wise_sampling`.
These four functions don't need to be implemented, they are just auxiliary functions to
:class:`~recbole.data.dataloader.abstract_dataloader.NegSampleDataLoader`.

In current studies, there have only two sampling strategies,
the first one is ``pair-wise sampling``, the other is ``point-wise sampling``.
:meth:`~recbole.data.dataloader.abstract_dataloader.NegSampleDataLoader._neg_sample_by_pair_wise_sampling`,
and :meth:`~recbole.data.dataloader.abstract_dataloader.NegSampleDataLoader._neg_sample_by_point_wise_sampling`
are implemented according to these two sampling strategies.

:meth:`~recbole.data.dataloader.abstract_dataloader.NegSampleDataLoader._set_neg_sample_args` is used to
set the negative sampling args like the sampling strategies, sampling functions and so on.
:meth:`~recbole.data.dataloader.abstract_dataloader.NegSampleDataLoader._neg_sampling` is used for negative sampling,
which will generate negative items and invoke
:meth:`~recbole.data.dataloader.abstract_dataloader.NegSampleDataLoader._neg_sample_by_pair_wise_sampling`,
or :meth:`~recbole.data.dataloader.abstract_dataloader.NegSampleDataLoader._neg_sample_by_point_wise_sampling`
according to the sampling strategies.


Example
--------------------------
Here, we take :class:`~recbole.data.dataloader.user_dataloader.UserDataLoader` as the example,
this dataloader returns user id, which is leveraged to train the user representations.


Implement __init__()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:meth:`__init__` can be used to initialize some of the necessary parameters.
Here, we just need to record :attr:`uid_field` and generate :attr:`user_list` which contains all user ids.
And because of some training requirements, :attr:`shuffle` should be set to ``True``.

.. code:: python

    def __init__(self, config, dataset, sampler, shuffle=False):
        if shuffle is False:
            shuffle = True
            self.logger.warning('UserDataLoader must shuffle the data.')

        self.uid_field = dataset.uid_field
        self.user_list = Interaction({self.uid_field: torch.arange(dataset.user_num)})

        super().__init__(config, dataset, sampler, shuffle=shuffle)

Implement _init_batch_size_and_step()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Because :class:`~recbole.data.dataloader.user_dataloader.UserDataLoader` don't need negative sampling,
so the :attr:`batch_size` and :attr:`step` can be both set to :attr:`self.config['train_batch_size']`.

.. code:: python

    def _init_batch_size_and_step(self):
        batch_size = self.config['train_batch_size']
        self.step = batch_size
        self.set_batch_size(batch_size)

Implement pr_end() and _shuffle()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Since this dataloader only returns user id, these function can be implemented readily.

.. code:: python

    @property
    def pr_end(self):
        return len(self.user_list)

    def _shuffle(self):
        self.user_list.shuffle()

Implement _next_batch_data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This function only requires to return user id from :attr:`user_list`,
we just select corresponding slice of :attr:`user_list` and return this slice.

.. code:: python

    def _next_batch_data(self):
        cur_data = self.user_list[self.pr:self.pr + self.step]
        self.pr += self.step
        return cur_data


Complete Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    class UserDataLoader(AbstractDataLoader):
        """:class:`UserDataLoader` will return a batch of data which only contains user-id when it is iterated.

        Args:
            config (Config): The config of dataloader.
            dataset (Dataset): The dataset of dataloader.
            sampler (Sampler): The sampler of dataloader.
            shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

        Attributes:
            shuffle (bool): Whether the dataloader will be shuffle after a round.
                However, in :class:`UserDataLoader`, it's guaranteed to be ``True``.
        """

        dl_type = DataLoaderType.ORIGIN

        def __init__(self, config, dataset, sampler, shuffle=False):
            if shuffle is False:
                shuffle = True
                self.logger.warning('UserDataLoader must shuffle the data.')

            self.uid_field = dataset.uid_field
            self.user_list = Interaction({self.uid_field: torch.arange(dataset.user_num)})

            super().__init__(config, dataset, sampler, shuffle=shuffle)

        def _init_batch_size_and_step(self):
            batch_size = self.config['train_batch_size']
            self.step = batch_size
            self.set_batch_size(batch_size)

        @property
        def pr_end(self):
            return len(self.user_list)

        def _shuffle(self):
            self.user_list.shuffle()

        def _next_batch_data(self):
            cur_data = self.user_list[self.pr:self.pr + self.step]
            self.pr += self.step
            return cur_data


Other more complex Dataloader development can refer to the source code.
