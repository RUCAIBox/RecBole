Load Pre-trained Embedding
===========================
 
For users who want to use pre-trained user(item) embedding to train their model. We provide a simple way as following.

Firstly, prepare your additional embedding feature file, which contain at least two columns (id & embedding vector) as following format and name it as ``dataset.suffix`` (e.g: ``ml-1m.useremb``).

=============   ===============================
uid:token           user_emb:float_seq
=============   ===============================
1               -115.08 13.60 113.69
2               -130.97 263.05 -129.88
=============   ===============================

Note that here the header of user id must be different from user id in your ``.user`` file or ``.inter`` file (e.g: if the header of user id in ``.user`` or ``.inter`` file is ``user_id:token``, the header of user id in your additional embedding feature file must be different. It can be either ``uid:token`` or ``userid:token``).

Secondly, update the args as (suppose that ``USER_ID_FIELD: user_id``):
 
.. code:: yaml

    additional_feat_suffix: [useremb]
    load_col:
        # inter/user/item/...: As usual
        useremb: [uid, user_emb]
    alias_of_user_id: [uid]
    preload_weight: 
        uid: user_emb

Then, this additional embedding feature file will be loaded into the :class:`Dataset` object. These new features can be accessed as following:

.. code:: python

    dataset = create_dataset(config)
    print(dataset.useremb_feat)

In your model, user embedding matrix can be initialized by your pre-trained embedding vectors as following:

.. code:: python

    class YourModel(GeneralRecommender):
        def __init__(self, config, dataset):
            pretrained_user_emb = dataset.get_preload_weight('uid')
            self.user_embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_user_emb))

