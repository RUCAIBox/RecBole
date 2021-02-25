Customize Samplers
======================
Here we present how to develop a new sampler, and apply it into RecBole.
The new sampler is used when we need complex sampling method.

Here, we take the :class:`~recbole.sampler.sampler.KGSampler` as an example.


Create a New Sampler Class
-----------------------------
To begin with, we create a new sampler based on :class:`~recbole.sampler.sampler.AbstractSampler`:

.. code:: python

    from recbole.sampler import AbstractSampler
    class KGSampler(AbstractSampler):
        pass


Implement __init__()
-----------------------
Then, we implement :meth:`~recbole.sampler.sampler.KGSampler.__init__()`, in this method, we can flexibly define and initialize the parameters,
where we only need to invoke :obj:`super.__init__(distribution)`.

.. code:: python

    def __init__(self, dataset, distribution='uniform'):
        self.dataset = dataset

        self.hid_field = dataset.head_entity_field
        self.tid_field = dataset.tail_entity_field
        self.hid_list = dataset.head_entities
        self.tid_list = dataset.tail_entities

        self.head_entities = set(dataset.head_entities)
        self.entity_num = dataset.entity_num

        super().__init__(distribution=distribution)


Implement get_random_list()
------------------------------
We do not use the random function in python or numpy due to their lower efficiency.
Instead, we realize our own :meth:`~recbole.sampler.sampler.AbstractSampler.random` function, where the key method is to combine the random list with the pointer.
The pointer point to some element in the random list. When one calls :meth:`self.random`, the element is returned, and moves the pointer backward by one element.
If the pointer point to the last element, then it will return to the head of the element.

In :class:`~recbole.sampler.sampler.AbstractSampler`, the :meth:`~recbole.sampler.sampler.AbstractSampler.__init__` will call :meth:`~recbole.sampler.sampler.AbstractSampler.get_random_list`, and shuffle the results.
We only need to return a list including all the elements.

It should be noted ``0`` can be the token used for padding, thus one should remain this value.

Example code:

.. code:: python

    def get_random_list(self):
        if self.distribution == 'uniform':
            return list(range(1, self.entity_num))
        elif self.distribution == 'popularity':
            return list(self.hid_list) + list(self.tid_list)
        else:
            raise NotImplementedError('Distribution [{}] has not been implemented'.format(self.distribution))


Implement get_used_ids()
----------------------------
For negative sampling, we do not want to sample positive instance, this function is used to compute the positive sample.
The function will return numpy, and the index is the ID. The return value will be saved in :attr:`self.used_ids`.

Example code:

.. code:: python

    def get_used_ids(self):
        used_tail_entity_id = np.array([set() for i in range(self.entity_num)])
        for hid, tid in zip(self.hid_list, self.tid_list):
            used_tail_entity_id[hid].add(tid)
        return used_tail_entity_id


Implementing the sampling function
-----------------------------------
In :class:`~recbole.sampler.sampler.AbstractSampler`, we have implemented :meth:`~recbole.sampler.sampler.AbstractSampler.sample_by_key_ids` function,
where we have three parameters: :attr:`key_ids`, :attr:`num` and :attr:`used_ids`.
:attr:`Key_ids` is the candidate objective ID list, :attr:`num` is the number of samples, :attr:`used_ids` are the positive sample list.

In the function, we sample :attr:`num` instances for each element in :attr:`key_ids`. The function finally return :class:`numpy.ndarray`,
the index of 0, len(key_ids), len(key_ids) * 2, …, len(key_ids) * (num - 1) is the result of key_ids[0].
The index of 1, len(key_ids) + 1, len(key_ids) * 2 + 1, …, len(key_ids) * (num - 1) + 1 is the result of key_ids[1].

One can also design her own sampler, if the above process is not appropriate.

Example code:

.. code:: python

    def sample_by_entity_ids(self, head_entity_ids, num=1):
        try:
            return self.sample_by_key_ids(head_entity_ids, num, self.used_ids[head_entity_ids])
        except IndexError:
            for head_entity_id in head_entity_ids:
                if head_entity_id not in self.head_entities:
                    raise ValueError('head_entity_id [{}] not exist'.format(head_entity_id))


Complete Code
----------------------
.. code:: python

    class KGSampler(AbstractSampler):
        """:class:`KGSampler` is used to sample negative entities in a knowledge graph.

        Args:
            dataset (Dataset): The knowledge graph dataset, which contains triplets in a knowledge graph.
            distribution (str, optional): Distribution of the negative entities. Defaults to 'uniform'.
        """
        def __init__(self, dataset, distribution='uniform'):
            self.dataset = dataset

            self.hid_field = dataset.head_entity_field
            self.tid_field = dataset.tail_entity_field
            self.hid_list = dataset.head_entities
            self.tid_list = dataset.tail_entities

            self.head_entities = set(dataset.head_entities)
            self.entity_num = dataset.entity_num

            super().__init__(distribution=distribution)

        def get_random_list(self):
            """
            Returns:
                np.ndarray or list: Random list of entity_id.
            """
            if self.distribution == 'uniform':
                return list(range(1, self.entity_num))
            elif self.distribution == 'popularity':
                return list(self.hid_list) + list(self.tid_list)
            else:
                raise NotImplementedError('Distribution [{}] has not been implemented'.format(self.distribution))

        def get_used_ids(self):
            """
            Returns:
                np.ndarray: Used entity_ids is the same as tail_entity_ids in knowledge graph.
                Index is head_entity_id, and element is a set of tail_entity_ids.
            """
            used_tail_entity_id = np.array([set() for i in range(self.entity_num)])
            for hid, tid in zip(self.hid_list, self.tid_list):
                used_tail_entity_id[hid].add(tid)
            return used_tail_entity_id

        def sample_by_entity_ids(self, head_entity_ids, num=1):
            """Sampling by head_entity_ids.

            Args:
                head_entity_ids (np.ndarray or list): Input head_entity_ids.
                num (int, optional): Number of sampled entity_ids for each head_entity_id. Defaults to ``1``.

            Returns:
                np.ndarray: Sampled entity_ids.
                entity_ids[0], entity_ids[len(head_entity_ids)], entity_ids[len(head_entity_ids) * 2], ...,
                entity_id[len(head_entity_ids) * (num - 1)] is sampled for head_entity_ids[0];
                entity_ids[1], entity_ids[len(head_entity_ids) + 1], entity_ids[len(head_entity_ids) * 2 + 1], ...,
                entity_id[len(head_entity_ids) * (num - 1) + 1] is sampled for head_entity_ids[1]; ...; and so on.
            """
            try:
                return self.sample_by_key_ids(head_entity_ids, num, self.used_ids[head_entity_ids])
            except IndexError:
                for head_entity_id in head_entity_ids:
                    if head_entity_id not in self.head_entities:
                        raise ValueError('head_entity_id [{}] not exist'.format(head_entity_id))

