Training Settings
===========================
Training settings are designed to set parameters about model training.


- ``epochs (int)`` : The number of training epochs. Defaults to ``300``.
- ``train_batch_size (int)`` : The training batch size. Defaults to ``2048``.
- ``learner (str)`` : The name of used optimizer. Defaults to ``'adam'``.
  Range in ``['adam', 'sgd', 'adagrad', 'rmsprop', 'sparse_adam']``.
- ``learning_rate (float)`` : Learning rate. Defaults to ``0.001``.
- ``train_neg_sample_args (dict)`` : This parameter have 4 keys: ``distribution``, ``sample_num``, ``dynamic``, and ``candidate_num``.   

  - ``distribution (str)`` : decides the distribution of negative items in sampling pools. Now we support two kinds of distribution: ``['uniform', 'popularity']``. ``uniform`` means uniformly select negative items while ``popularity`` means select negative items based on their popularity (Counter(item) in `.inter` file). The default value is ``uniform``.   

  - ``sample_num (int)`` : decides the number of negative samples we intend to take. The default value is ``1``.
  
  - ``dynamic (bool)`` : decides whether we adopt dynamic negative sampling. The default value is ``False``.
  
  - ``candidate_num (int)`` : decides the number of candidate negative items when dynamic negative sampling. The default value is ``0``.
- ``eval_step (int)`` : The number of training epochs before an evaluation
  on the valid dataset. If it is less than 1, the model will not be
  evaluated on the valid dataset. Defaults to ``1``.
- ``stopping_step (int)`` : The threshold for validation-based early stopping.
  Defaults to ``10``.
- ``clip_grad_norm (dict)`` : The args of `clip_grad_norm_ <https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>`_
  which will clip gradient norm of model. Defaults to ``None``.
- ``loss_decimal_place(int)``: The decimal place of training loss. Defaults to ``4``.
- ``weight_decay (float)`` : The weight decay (L2 penalty), used for `optimizer <https://pytorch.org/docs/stable/optim.html?highlight=weight_decay>`_. Default to ``0.0``.
- ``require_pow (bool)``: The sign identifies whether the power operation is performed based on the norm in EmbLoss. Defaults to ``False``.
- ``enable_amp (bool)``: The parameter determines whether to use mixed precision training. Defaults to ``False``.
- ``enable_scaler (bool)``: The parameter determines whether to use GradScaler that is often used with mixed precision training to avoid gradient precision overflow. Defaults to ``False``.
