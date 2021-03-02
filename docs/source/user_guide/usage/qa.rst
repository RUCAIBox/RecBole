Clarifications on some practical issues
=========================================

**Q1**:

Why the result of ``Dataset.item_num`` always one plus of the actual number of items in the dataset?

**A1**:

We add ``[PAD]`` for all the token like fields. Thus after remapping ID, ``0`` will be reserved for ``[PAD]``, which makes the result of ``Dataset.item_num`` more than the actual number.

Note that for Knowledge-based models, we add one more relation called ``U-I Relation``. It describes the history interactions which will be used in :meth:`recbole.data.dataset.kg_dataset.KnowledgeBasedDataset.ckg_graph`.
Thus the result of ``KGDataset.relation_num`` is two more than the actual number of relations.

**Q2**:

Why are the test results usually better than the best valid results?

**A2**:

For more rigorous evaluation, those user-item interaction records in validation sets will not be ranked while testing.
Thus the distribution of validation & test sets may be inconsistent.

However, this doesn't affect the comparison between models.

**Q3**

Why do I receive a warning about ``batch_size changed``? What is the meaning of :attr:`batch_size` in dataloader?

**A3**

In RecBole's dataloader, the meaning of :attr:`batch_size` is the upper bound of the number of **interactions** in one single batch.

On the one hand, it's easy to calculate and control the usage of GPU memories. E.g., while comparing between different datasets, you don't need to change the value of :attr:`batch_size`, because the usage of GPU memories will not change a lot.

On the other hand, in RecBole's top-k evaluation, we need the interactions of each user grouped in one batch. In other words, the interactions of any user should not be separated into multiple batches. We try to feed more interactions into one batch, but due to the above rules, the :attr:`batch_size` is just an upper bound. And :meth:`_batch_size_adaptation` is designed to adapt the actual batch size dynamically. Thus, while executing :meth:`_batch_size_adaptation`, you will receive a warning message.
