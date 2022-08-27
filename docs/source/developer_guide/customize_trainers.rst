Customize Trainers
======================
Here, we present how to develop a new Trainer, and apply it into RecBole.
For a new model, if the training method is complex, and existing trainer can not be used for training and evaluation,
then we need to develop a new trainer.

The function used to train the model is :meth:`fit`, it will call :meth:`_train_epoch` to train the model.

The function used to evaluate the model is :meth:`evaluate`, it will call :meth:`_valid_epoch` to evaluate the model.

If the developed model needs more complex training method,
then one can inherent the :class:`~recbole.trainer.trainer.Trainer`,
and revise :meth:`~recbole.trainer.trainer.Trainer.fit` or :meth:`~recbole.trainer.trainer.Trainer._train_epoch`.

If the developed model needs more complex evaluation method,
then one can inherent the :class:`~recbole.trainer.trainer.Trainer`,
and revise :meth:`~recbole.trainer.trainer.Trainer.evaluate` or :meth:`~recbole.trainer.trainer.Trainer._valid_epoch`.


Example
----------------
1. Alternative Optimization
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Here we present a simple Trainer example, which is used for alternative optimization.
We revise the :meth:`~recbole.trainer.trainer.Trainer._train_epoch` method.
To begin with, we need to create a new class for
:class:`NewTrainer` based on :class:`~recbole.trainer.trainer.Trainer`.

.. code:: python

    from recbole.trainer import Trainer

    class NewTrainer(Trainer):

        def __init__(self, config, model):
            super(NewTrainer, self).__init__(config, model)


Then we revise :meth:`~recbole.trainer.trainer.Trainer._train_epoch`.
Here, the losses are alternatively optimized after each epoch,
and the losses are computed by :meth:`calculate_loss1` and :meth:`calculate_loss2`


.. code:: python

    def _train_epoch(self, train_data, epoch_idx):
        self.model.train()
        total_loss = 0.

        if epoch_idx % 2 == 0:
            for batch_idx, interaction in enumerate(train_data):
                interaction = interaction.to(self.device)
                self.optimizer.zero_grad()
                loss = self.model.calculate_loss1(interaction)
                self._check_nan(loss)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        else:
            for batch_idx, interaction in enumerate(train_data):
                interaction = interaction.to(self.device)
                self.optimizer.zero_grad()
                loss = self.model.calculate_loss2(interaction)
                self._check_nan(loss)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        return total_loss


Complete Code
^^^^^^^^^^^^^^^^

.. code:: python

    from recbole.trainer import Trainer

    class NewTrainer(Trainer):

        def __init__(self, config, model):
            super(NewTrainer, self).__init__(config, model)

        def _train_epoch(self, train_data, epoch_idx):
            self.model.train()
            total_loss = 0.

            if epoch_idx % 2 == 0:
                for batch_idx, interaction in enumerate(train_data):
                    interaction = interaction.to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.model.calculate_loss1(interaction)
                    self._check_nan(loss)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
            else:
                for batch_idx, interaction in enumerate(train_data):
                    interaction = interaction.to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.model.calculate_loss2(interaction)
                    self._check_nan(loss)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
            return total_loss

2. Mixed precision training
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Here we present a simple Trainer example, which is used for mixed
precision training. Mixed precision training offers significant
computational speedup by performing operations in half-precision
format, while storing minimal information in single-precision to
retain as much information as possible in critical parts of the
network. Let's give an example based on torch ``torch.autocast``. To
begin with, we need to create a new class for ``NewTrainer`` based on
``Trainer``.

.. code:: python

  from recbole.trainer import Trainer
  import torch.cuda.amp as amp 
  class NewTrainer(Trainer):
      def __init__(self, config, model):
          super(NewTrainer, self).__init__(config, model)
          
Then we revise ``_train_epoch()``.

.. code:: python

  def _train_epoch(self, train_data, epoch_idx):
      self.model.train()
      scaler = amp.GradScaler(enabled=self.enable_scaler)
      for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                losses = loss_func(interaction)
            total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
Complete Code
^^^^^^^^^^^^^^^^
.. code:: python

  from recbole.trainer import Trainer
  import torch.cuda.amp as amp 
  class NewTrainer(Trainer):
      def __init__(self, config, model):
          super(NewTrainer, self).__init__(config, model)
          
  def _train_epoch(self, train_data, epoch_idx):
      self.model.train()
      scaler = amp.GradScaler(enabled=self.enable_scaler)
      for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                losses = loss_func(interaction)
            total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()        

There are some points to note: 

1. ``GradScaler`` can only be used on GPU, while ``torch.autocast`` can be used both on CPU and GPU.

2.  Some models whose loss value is too large will cause overflow (e.g., Caser, CDAE,DIEN),
    and these models are not suitable for mixed precision training.
    If you see errors like "RuntimeError: Function 'xxx' returned nan values",
    please disable mixed precision training by setting ``enable_amp`` and ``enable_scaler`` to False.

3.  Because pytorch does not support single-precision sparse matrix multiplication, models using ``torch.sparse.mm``, 
    including NGCF, DMF, GCMC, LightGCN, NCL, SGL, SpectralCF and KGAT cannot be trained with mixed precision.


3. Layer-specific learning rate
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Here we present a simple Trainer example, which is used for setting
layer-specific learning rate. For pretrained model, layers closer to
the input layer are more likely to have learned more general
features. On the other hand, later layers of the model learn the
detailed features. In this case, we can set different learning rate
for different layers. We can do this by modifying the optimizer.

.. code:: python

      def _build_optimizer(self, learner, learning_rate, weight_decay):
          pretrained_params = list(map(id, self.model.pretrained_part.parameters())
          base_params = filter(lambda p: id(p) not in pretrained_params, self.model.parameters())
          if learner.lower() == 'adam':
              optimizer = optim.Adam([
                  {"params":base_params},
                  {"pretrained_params":self.model.pretrained_part.parameters(),"lr":1e-5}],
                  lr=learning_rate,weight_decay=weight_decay)
          return optimizer             



Complete Code
^^^^^^^^^^^^^^^^
.. code:: python 

  from recbole.trainer import Trainer
  class NewTrainer(Trainer):
      def __init__(self, config, model):
          super(NewTrainer, self).__init__(config, model)
          self.optimizer = self._build_optimizer()
          
  def _train_epoch(self, train_data, epoch_idx):
          self.model.train()
          total_loss = 0.
          for batch_idx, interaction in enumerate(train_data):
          interaction = interaction.to(self.device)
          self.optimizer.zero_grad()
          loss = self.model.calculate_loss1(interaction)
          self._check_nan(loss)
          loss.backward()
          self.optimizer.step()
          total_loss += loss.item()
          return total_loss
