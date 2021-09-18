Customize Samplers
======================
In RecBole, sampler module is designed to select negative items for training and evaluation.

Here we present how to develop a new sampler, and apply it into RecBole.
The new sampler is used when we need complex sampling method.

In RecBole, we now only support two kinds of sampling strategies: **random negative sampling (RNS)** and **popularity-biased negative sampling (PNS)**.
RNS is to select the negative items in uniform distribution, and PNS is to select the negative item in a popularity-biased distribution. 
For PNS, we set the popularity-biased distribution based on the total number of items' interactions.

In our framework, if you want to create a new sampler, you need to inherit the :class:`~recbole.sampler.sampler.AbstractSampler`, implement
:meth:`~recbole.sampler.sampler.AbstractSampler.__init__`,
rewrite three functions: :meth:`~recbole.sampler.sampler.AbstractSampler._uni_sampling`,
:meth:`~recbole.sampler.sampler.AbstractSampler._get_candidates_list`,
:meth:`~recbole.sampler.sampler.AbstractSampler.get_used_ids`
and create a new sampling function.


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
Then, we implement :meth:`~recbole.sampler.sampler.KGSampler.__init__`, in this method, we can flexibly define and initialize the parameters,
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

Implement _uni_sampling()
-------------------------------
To implement the RNS for KGSampler, we need to rewrite the :meth:`~recbole.sampler.sampler.AbstractSampler._uni_sampling`.
Here we use the :meth:`numpy.random.randint` to help us randomly select the ``entity_id``. This function will return the
selected samples' id (here is ``entity_id``).

Example code:

.. code:: python

    def _uni_sampling(self, sample_num):
        return np.random.randint(1, self.entity_num, sample_num)

Implement _get_candidates_list()
-------------------------------------
To implement PNS for KGSampler, we need to rewrite the :meth:`~recbole.sampler.sampler.AbstractSampler._get_candidates_list`.
This function is used to get a candidate list for PNS, and we will set the sampling distribution based on 
:obj:`Counter(candidate_list)`. This function will return a list of candidates' id.

Example code:

.. code:: python

    def _get_candidates_list(self):
        return list(self.hid_list) + list(self.tid_list)


Implement get_used_ids()
----------------------------
For negative sampling, we do not want to sample positive instance, this function is used to record the positive sample.
The function will return numpy, and the index is the ID. The returned value will be saved in :attr:`self.used_ids`.

Example code:

.. code:: python

    def get_used_ids(self):
       used_tail_entity_id = np.array([set() for _ in range(self.entity_num)])
        for hid, tid in zip(self.hid_list, self.tid_list):
            used_tail_entity_id[hid].add(tid)

        for used_tail_set in used_tail_entity_id:
            if len(used_tail_set) + 1 == self.entity_num:  # [pad] is a entity.
                raise ValueError(
                    'Some head entities have relation with all entities, '
                    'which we can not sample negative entities for them.'
                )
        return used_tail_entity_id


Implement the sampling function
-----------------------------------
In :class:`~recbole.sampler.sampler.AbstractSampler`, we have implemented :meth:`~recbole.sampler.sampler.AbstractSampler.sample_by_key_ids` function,
where we have three parameters: :attr:`key_ids`, :attr:`num` and :attr:`used_ids`.
:attr:`Key_ids` is the candidate objective ID list, :attr:`num` is the number of samples, :attr:`used_ids` is the positive sample list.

In the function, we sample :attr:`num` instances for each element in :attr:`key_ids`. The function finally return :class:`numpy.ndarray`,
the index of 0, len(key_ids), len(key_ids) * 2, …, len(key_ids) * (num - 1) is the result of key_ids[0].
The index of 1, len(key_ids) + 1, len(key_ids) * 2 + 1, …, len(key_ids) * (num - 1) + 1 is the result of key_ids[1].

One can also design his own sampler, if the above process is not appropriate.

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
        def __init__(self, dataset, distribution='uniform'):
            self.dataset = dataset

            self.hid_field = dataset.head_entity_field
            self.tid_field = dataset.tail_entity_field
            self.hid_list = dataset.head_entities
            self.tid_list = dataset.tail_entities

            self.head_entities = set(dataset.head_entities)
            self.entity_num = dataset.entity_num

            super().__init__(distribution=distribution)

        def _uni_sampling(self, sample_num):
            return np.random.randint(1, self.entity_num, sample_num)

        def _get_candidates_list(self):
            return list(self.hid_list) + list(self.tid_list)

        def get_used_ids(self):
            used_tail_entity_id = np.array([set() for _ in range(self.entity_num)])
            for hid, tid in zip(self.hid_list, self.tid_list):
                used_tail_entity_id[hid].add(tid)

            for used_tail_set in used_tail_entity_id:
                if len(used_tail_set) + 1 == self.entity_num:  # [pad] is a entity.
                    raise ValueError(
                        'Some head entities have relation with all entities, '
                        'which we can not sample negative entities for them.'
                    )
            return used_tail_entity_id

        def sample_by_entity_ids(self, head_entity_ids, num=1):
            try:
                return self.sample_by_key_ids(head_entity_ids, num)
            except IndexError:
                for head_entity_id in head_entity_ids:
                    if head_entity_id not in self.head_entities:
                        raise ValueError(f'head_entity_id [{head_entity_id}] not exist.')


