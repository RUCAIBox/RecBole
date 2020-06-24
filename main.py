from config import Config
from dataset import Dataset
from model import Model
from trainer import Trainer
from evaluator import Evaluator



config = Config()
config.init()


dataset = Dataset(config)
train_data, test_data = dataset.preprocessing()


model = Model(config)
trainer = Trainer(config)
model = trainer.train(model, train_data)
result = trainer.predict(model, test_data)


evaluator = Evaluator(config)
evaluator.evaluate(result, test_data)
