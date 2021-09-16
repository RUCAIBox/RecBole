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
