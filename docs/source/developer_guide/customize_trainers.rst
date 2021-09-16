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

