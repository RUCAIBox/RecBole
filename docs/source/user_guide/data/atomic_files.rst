Atomic Files
===================

Atomic files are introduced to format the input of mainstream recommendation tasks in a flexible way.

So far, our library introduces six atomic file types, and we identify different files by their suffixes.

=========  ==============================  ========================================================
Suffix        Content                             Example Format
=========  ==============================  ========================================================
`.inter`   User-item interaction             `user_id`, `item_id`, `rating`, `timestamp`, `review`
`.user`    User feature                      `user_id`, `age`, `gender`
`.item`    Item feature                      `item_id`, `category`
`.kg`      Triplets in a knowledge graph     `head_entity`, `tail_entity`, `relation`
`.link`    Item-entity linkage data          `entity`, `item_id`
`.net`     Social graph data                 `source`, `target`
=========  ==============================  ========================================================

Atomic files are combined to support the input of different recommendation tasks.

One can write the suffixes into the config arg ``load_col`` to load the corresponding atomic files.

For each recommendation task, we have to provide several mandatory files:

================              ================================
Tasks                             Mandatory atomic files
================              ================================
General                         `.inter`
Context-aware                   `.inter`, `.user`, `.item`
Knowledge-aware                 `.inter`, `.kg`, `.link`
Sequential                      `.inter`
Social                          `.inter`, `.net`
================              ================================

Format
--------

Each atomic file can be viewed as a m x n table, where n is the number of features and m-1 is the number of data records(one line for header).

The first row corresponds to feature names, in which each entry has the form of ``feat_name:feat_type``，indicating the feature name and feature type.

We support four feature types, which can be processed by tensors in batch.

============   ===========================   =====================
feat_type        Explanations                 Examples
============   ===========================   =====================
`token`        single discrete feature        `user_id`, `age`
`token_seq`    discrete features sequence     `review`
`float`        single continuous feature      `rating`, `timestamp`
`float_seq`    continuous feature sequence    `vector`
============   ===========================   =====================

Examples
----------

We present three example data rows in the formatted ML-1M dataset.

**ml-1m.inter**

=============   =============   ============   ===============
user_id:token   item_id:token   rating:float   timestamp:float
=============   =============   ============   ===============
1               1193            5              978300760
1               661             3              978302109
=============   =============   ============   ===============

**ml-1m.user**

=============   =========   ============   ================   ==============
user_id:token   age:token   gender:token   occupation:token   zip_code:token
=============   =========   ============   ================   ==============
1               1           F              10                 48067
2               56          M              16                 70072
=============   =========   ============   ================   ==============

**ml-1m.item**

=============   =====================   ==================   ============================
item_id:token   movie_title:token_seq   release_year:token   genre:token_seq
=============   =====================   ==================   ============================
1               Toy Story               1995                 Animation Children's Comedy
2               Jumanji                 1995                 Adventure Children's Fantasy
=============   =====================   ==================   ============================

**ml-1m.kg**

=============   ===================================   =============
head_id:token   relation_id:token                     tail_id:token
=============   ===================================   =============
m.0gs6m         film.film_genre.films_in_this_genre   m.01b195
m.052_dz        film.film.actor                       m.02nrdp
=============   ===================================   =============

**ml-1m.link**

=============   ===============
item_id:token   entity_id:token
=============   ===============
2694            m.02hxhz
2079            m.0kvcr9
=============   ===============

Additional Atomic Files
----------------------------

For users who want to load features from additional atomic files (e.g. pretrained entity embeddings), we provide a simple way as following.

Firstly, prepare your additional atomic file (e.g. ``ml-1m.ent``).

=============   ===============================
ent_id:token    ent_emb:float_seq
=============   ===============================
m.0gs6m         -115.08 13.60 113.69
m.01b195        -130.97 263.05 -129.88
=============   ===============================

Secondly, update the args as:

.. code:: yaml

    additional_feat_suffix: [ent]
    load_col:
        # inter/user/item/...: As usual
        ent: [ent_id, ent_emb]

Then, this additional atomic file will be loaded into the :class:`Dataset` object. These new features can be used as following.

.. code:: python

    dataset = create_dataset(config)
    print(dataset.ent_feat)

Note that these features can be preprocessed by the same way as the other features.

For example, if you want to map the tokens of ``ent_id`` into the same space of ``entity_id``, then update the args as:

.. code:: yaml

    additional_feat_suffix: [ent]
    load_col:
        # inter/user/item/...: As usual
        ent: [ent_id, ent_emb]

    alias_of_entity_id: [ent_id]
