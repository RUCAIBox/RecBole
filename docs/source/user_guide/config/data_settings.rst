Data settings
=========================

RecBole provides several arguments for describing:

- Basic information of the dataset
- Operations of dataset preprocessing

See below for the details:

Atomic File Format
----------------------

- ``field_separator (str)`` : Separator of different columns in atomic files. Defaults to ``"\t"``.
- ``seq_separator (str)`` : Separator inside the sequence features. Defaults to ``" "``.

Basic Information
----------------------

Common Features
''''''''''''''''''

- ``USER_ID_FIELD (str)`` : Field name of user ID feature. Defaults to ``user_id``.
- ``ITEM_ID_FIELD (str)`` : Field name of item ID feature. Defaults to ``item_id``.
- ``RATING_FIELD (str)`` : Field name of rating feature. Defaults to ``rating``.
- ``TIME_FIELD (str)`` : Field name of timestamp feature. Defaults to ``timestamp``.
- ``seq_len (dict)`` : Keys are field names of sequence features, values are maximum length of each sequence (which means too long sequences will be cut off). If not set, the sequences will not be cut off. Defaults to ``None``.

Label for Point-wise DataLoader
'''''''''''''''''''''''''''''''''''

- ``LABEL_FIELD (str)`` : Expected field name of the generated labels. Defaults to ``label``.
- ``threshold (dict)`` : The format is ``{k (str): v (float)}``. 0/1 labels will be generated according to the value of ``inter_feat[k]`` and ``v``. The rows with ``inter_feat[k] >= v`` will be labeled as positive, otherwise the label is negative. Note that at most one pair of ``k`` and ``v`` can exist in ``threshold``. Defaults to ``None``.

NegSample Prefix for Pair-wise DataLoader
''''''''''''''''''''''''''''''''''''''''''''''''''

- ``NEG_PREFIX (str)`` : Prefix of field names which are generated as negative cases. E.g. if we have positive item ID named ``item_id``, then those item ID in negative samples will be called ``NEG_PREFIX + item_id``. Defaults to ``neg_``.

Sequential Model Needed
'''''''''''''''''''''''''''''''''''

- ``ITEM_LIST_LENGTH_FIELD (str)`` : Field name of the feature representing item sequences' length. Defaults to ``item_length``.
- ``LIST_SUFFIX (str)`` : Suffix of field names which are generated as sequences. E.g. if we have item ID named ``item_id``, then those item ID sequences will be called ``item_id + LIST_SUFFIX``. Defaults to ``_list``.
- ``MAX_ITEM_LIST_LENGTH (int)``: Maximum length of each generated sequence. Defaults to ``50``.
- ``POSITION_FIELD (str)`` : Field name of the generated position sequence. For sequence of length ``k``, its position sequence is ``range(k)``. Note that this field will only be generated if this arg is not ``None``. Defaults to ``position_id``.

Knowledge-based Model Needed
'''''''''''''''''''''''''''''''''''

- ``HEAD_ENTITY_ID_FIELD (str)`` : Field name of the head entity ID feature. Defaults to ``head_id``.
- ``TAIL_ENTITY_ID_FIELD (str)`` : Field name of the tail entity ID feature. Defaults to ``tail_id``.
- ``RELATION_ID_FIELD (str)`` : Field name of the relation ID feature. Defaults to ``relation_id``.
- ``ENTITY_ID_FIELD (str)`` : Field name of the entity ID. Note that it's only a symbol of entities, not real feature of one of the ``xxx_feat``. Defaults to ``entity_id``.

Selectively Loading
------------------------------

- ``load_col (dict)`` : Keys are the suffix of loaded atomic files, values are the list of field names to be loaded. If a suffix doesn't exist in ``load_col``, the corresponding atomic file will not be loaded. Note that if ``load_col`` is ``None``, then all the existed atomic files will be loaded. Defaults to ``{inter: [user_id, item_id]}``.
- ``unload_col (dict)`` : Keys are suffix of loaded atomic files, values are list of field names NOT to be loaded. Note that ``load_col`` and ``unload_col`` can not be set at the same time. Defaults to ``None``.
- ``unused_col (dict)`` : Keys are suffix of loaded atomic files, values are list of field names which is loaded for data processing but will not be used in model. E.g. the ``time_field`` may be used for time ordering but model does not use this field. Defaults to ``None``.
- ``additional_feat_suffix (list)``: Control loading additional atomic files. E.g. if you want to load features from ``ml-100k.hello``, just set this arg as ``additional_feat_suffix: [hello]``. Features of additional features will be stored in ``Dataset.feat_list``. Defaults to ``None``.

Filtering
-----------

Remove duplicated user-item interactions
''''''''''''''''''''''''''''''''''''''''

- ``rm_dup_inter (str)`` : Whether to remove duplicated user-item interactions. If ``time_field`` exists, ``inter_feat`` will be sorted by ``time_field`` in ascending order. Otherwise it will remain unchanged. After that, if ``rm_dup_inter ==  first``, we will keep the first user-item interaction in duplicates; if ``rm_dup_inter == last``, we will keep the last user-item interaction in duplicates. Defaults to ``None``.

Filter by value
''''''''''''''''''

- ``val_interval (dict)``: Has the format ``{k (str): interval (str), ...}``, where ``interval`` can be set as ``[A,B]`` / ``[A,B)`` / ``(A,B)`` / ``(A,B]``. The rows whose ``feat[k]`` is in the interval ``interval`` will be retained. If you want to specify more than one interval, separate them with semicolon(s). For instance, ``{k: "[A,B);(C,D]"}`` can be adopted and rows whose ``feat[k]`` is in any specified interval will be retained. Defaults to ``None``, which means all rows will be retained.

Remove interation by user or item
'''''''''''''''''''''''''''''''''''

- ``filter_inter_by_user_or_item (bool)`` : If ``True``, we will remove the interaction in ``inter_feat`` which user or item is not in ``user_feat`` or ``item_feat``. Defaults to ``True``.

Filter by number of interactions
''''''''''''''''''''''''''''''''''''

- ``user_inter_num_interval (str)`` : Has the interval format, such as ``[A,B]`` / ``[A,B)`` / ``(A,B)`` / ``(A,B]``,  where ``A`` and ``B`` are the endpoints of the interval and ``A <= B``. Users whose number of interactions is in the interval will be retained. Defaults to ``[0,inf)``.
- ``item_inter_num_interval (str)`` : Has the interval format, such as ``[A,B]`` / ``[A,B)`` / ``(A,B)`` / ``(A,B]``,  where ``A`` and ``B`` are the endpoints of the interval and ``A <= B``. Items whose number of interactions is in the interval will be retained. Defaults to ``[0,inf)``.

Preprocessing
-----------------

- ``alias_of_user_id (list)``: List of fields' names, which will be remapped into the same index system with ``USER_ID_FIELD``. Defaults to ``None``.
- ``alias_of_item_id (list)``: List of fields' names, which will be remapped into the same index system with ``ITEM_ID_FIELD``. Defaults to ``None``.
- ``alias_of_entity_id (list)``: List of fields' names, which will be remapped into the same index system with ``ENTITY_ID_FIELD``, ``HEAD_ENTITY_ID_FIELD`` and ``TAIL_ENTITY_ID_FIELD``. Defaults to ``None``.
- ``alias_of_relation_id (list)``: List of fields' names, which will be remapped into the same index system with ``RELATION_ID_FIELD``. Defaults to ``None``.
- ``preload_weight (dict)`` : Has the format ``{k (str): v (float)}, ...``. ``k`` is a token field, representing the IDs of each row of preloaded weight matrix. ``v`` is a float-like field. Each pair of ``k`` and ``v`` should be from the same atomic file. This arg can be used to load pretrained vectors. Defaults to ``None``.
- ``normalize_field (list)`` : List of filed names to be normalized. Note that only float-like fields can be normalized. Defaults to ``None``.
- ``normalize_all (bool)`` : Normalize all the float like fields if ``True``. Defaults to ``None``.

Benchmark file
-------------------

- ``benchmark_filename (list)`` : List of pre-split user-item interaction suffix. We will only apply normalize, remap-id, which will not delete the interaction in inter_feat. And then split the inter_feat by ``benchmark_filename``. E.g. Let's assume that the dataset is called ``click``, and ``benchmark_filename`` equals to ``['part1', 'part2', 'part3']``. That we will load ``click.part1.inter``, ``click.part2.inter``, ``click.part3.inter``, and treat them as train, valid, test dataset. Defaults to ``None``.
