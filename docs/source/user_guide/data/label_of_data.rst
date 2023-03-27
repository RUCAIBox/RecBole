Label of data
=========================
In recommendation filed, there are two kinds of data scenes: explicit feedback scene and implicit feedback scene. 

Explicit feedback, like rating for items, has explicit label for model training. While for implicit feedback, like clicks and purchases, 
the label of data is vague, and generally we will regard all the observed interaction as the positive samples and select negative samples from
unobserved interactions (known as negative sampling).    

To supports both explicit feedback scene and implicit feedback scene, RecBole design three ways to set label of data.

1. Set label field
-----------------------------
If your data has already been labeled, you only need to set ``LABEL_FIELD`` to tell the model 
which column represents the label of data, and then set `train_neg_sample_args` as `None`.

For example, if your `.inter` file is like:

=============   =============   ============   ===============
user_id:token   item_id:token   label:float    timestamp:float
=============   =============   ============   ===============
1               1193            1              978300760
1               661             0              978302109
2               11              1              978302009  
2               112             1              978312344 
2               555             0              978302321 
3               234             1              978302109 
=============   =============   ============   ===============

Then, you can set the config like:

.. code:: yaml

    LABEL_FIELD: label
    train_neg_sample_args: None

Note that the value of your label column should only be 0 or 1 (0 represents the negative label and 
1 represents the positive label). 

2. Set threshold
------------------------------

If your data doesn't have labels but has users' feedback information (like rating for items) to show their preferences, 
a general way to label them is to set threshold. 

For example, if you `.inter` file is like:

=============   =============   ============   ===============
user_id:token   item_id:token   rating:float   timestamp:float
=============   =============   ============   ===============
1               1193            5              978300760
1               661             1              978302109
2               11              4              978302009  
2               112             4              978312344 
2               555             1              978302321 
3               234             3              978302109 
=============   =============   ============   ===============

To set label for these interactions, you can set `3` as the threshold of rating, and 
the interactions will be labeled as positive if their rating no less than 3.

You can set the config like:

.. code:: yaml

    threshold: 
        rating: 3
    train_neg_sample_args: None

And then RecBole will automatically set label for interactions based on their rating column. 
    
3. Negative sampling
------------------------------
If your only have implicit feedback data, without label or users' feedback information.
A general way to label these kinds of data is negative sampling. We will assume that for each user, all the observed interactions are positive,
and the unobserved ones are negative. And then, we will set positive label for all the observed interactions, 
and select some negative samples from the unobserved interactions according to a certain strategy.

You can set the config like:

.. code:: yaml

    train_neg_sample_args: 
        uniform: 1

And then, RecBole will automatically select one negative sample for each positive sample uniformly from the unobserved interactions.

At last, for more details about the label config, please read :doc:`../config/data_settings` and :doc:`../config/training_settings`.



