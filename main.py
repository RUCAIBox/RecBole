from config import Config
from data import Dataset
from model.general_recommender.bprmf import BPRMF
from trainer import Trainer
from utils import Logger

config = Config('properties/overall.config')
config.init()

logger = Logger(config)

dataset = Dataset(config)

train_data, test_data, valid_data = dataset.build()

model = BPRMF(config, dataset).to(config['device'])

trainer = Trainer(config, model, logger)
# trainer.resume_checkpoint('saved/model_best.pth')
trainer.fit(train_data, valid_data)
result = trainer.evaluate(test_data)
print(result)
trainer.plot_train_loss(show=True)
