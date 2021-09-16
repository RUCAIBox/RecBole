Training Settings
===========================
Training settings are designed to set parameters about model training.


- ``epochs (int)`` : The number of training epochs. Defaults to ``300``.
- ``train_batch_size (int)`` : The training batch size. Defaults to ``2048``.
- ``learner (str)`` : The name of used optimizer. Defaults to ``'adam'``.
  Range in ``['adam', 'sgd', 'adagrad', 'rmsprop', 'sparse_adam']``.
- ``learning_rate (float)`` : Learning rate. Defaults to ``0.001``.
- ``neg_sampling(dict)``: This parameter controls the negative sampling for model training.
  The key range is ``['uniform', 'popularity']``, which decides the distribution of negative items in sampling pools.
  ``uniform`` means uniformly select negative items while ``popularity`` means select negative items based on 
  their popularity (Counter(item) in `.inter` file). Note that if your data is labeled, you need to set this parameter as ``None``.
  The default value of this parameter is ``{'uniform': 1}``. 
- ``eval_step (int)`` : The number of training epochs before an evaluation
  on the valid dataset. If it is less than 1, the model will not be
  evaluated on the valid dataset. Defaults to ``1``.
- ``stopping_step (int)`` : The threshold for validation-based early stopping.
  Defaults to ``10``.
- ``clip_grad_norm (dict)`` : The args of `clip_grad_norm_ <https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>`_
  which will clip gradient norm of model. Defaults to ``None``.
- ``loss_decimal_place(int)``: The decimal place of training loss. Defaults to ``4``.
- ``weight_decay (float)`` : The weight decay (L2 penalty), used for `optimizer <https://pytorch.org/docs/stable/optim.html?highlight=weight_decay>`_. Default to ``0.0``.