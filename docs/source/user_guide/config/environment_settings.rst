Environment Settings
===========================
Environment settings are designed to set basic parameters of running environment.

- ``gpu_id (str)`` : The id of available GPU device(s). Defaults to ``0``.
- ``worker (int)`` : The number of workers processing the data. Defaults to ``0``.
- ``seed (int)`` : Random seed. Defaults to ``2020``.
- ``state (str)`` : Logging level. Defaults to ``'INFO'``.
  Range in ``['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL']``.
- ``encoding (str)``: Encoding to use for reading atomic files. Defaults to ``'utf-8'``.
  The available encoding can be found in `here <https://docs.python.org/3/library/codecs.html#standard-encodings>`__.
- ``reproducibility (bool)`` : If True, the tool will use deterministic
  convolution algorithms, which makes the result reproducible. If False,
  the tool will benchmark multiple convolution algorithms and select the fastest one,
  which makes the result not reproducible but can speed up model training in
  some case. Defaults to ``True``.
- ``data_path (str)`` : The path of input dataset. Defaults to ``'dataset/'``.
- ``checkpoint_dir (str)`` : The path to save checkpoint file.
  Defaults to ``'saved/'``.
- ``show_progress (bool)`` : Whether or not to show the progress bar of training and evaluation epochs.
  Defaults to ``True``.
- ``save_dataset (bool)``: Whether or not to save filtered dataset.
  If True, save filtered dataset, otherwise it will not be saved.
  Defaults to ``False``.
- ``dataset_save_path (str)``: The path of saved dataset. The tool will attempt to load the dataset from this path.
  If it equals to ``None``, the tool will try to load the dataset from ``{checkpoint_dir}/{dataset}-{dataset_class_name}.pth``.
  If the config of saved dataset is not equal to current config, the tool will create dataset from scratch.
  Defaults to ``None``.
- ``save_dataloaders (bool)``: Whether or not to save split dataloaders.
  If True, save split dataloaders, otherwise they will not be saved.
  Defaults to ``False``.
- ``dataloaders_save_path (str)``: The path of saved dataloaders. The tool will attempt to load the dataloaders from this path.
  If it equals to ``None``, the tool will try to load the dataloaders from ``{checkpoint_dir}/{dataset}-for-{model}-dataloader.pth``.
  If the config of saved dataloaders is not equal to current config, the tool will create dataloaders from scratch.
  Defaults to ``None``.
- ``log_wandb (bool)``: Whether or not to use Weights & Biases(W&B).
  If True, use W&B to visualize configs and metrics of different experiments, otherwise it will not be used.
  Defaults to ``False``.
- ``wandb_project (str)``: The project to conduct experiments in W&B.
  Defaults to ``'recbole'``.
- ``shuffle (bool)``: Whether or not to shuffle the training data before each epoch. Defaults to ``True``.