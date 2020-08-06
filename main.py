from config import Config
from data import Dataset, data_preparation
from model.general_recommender.bprmf import BPRMF
from trainer import Trainer
from utils import get_logger

config = Config('properties/overall.config')
config.init()
logger = get_logger(config)

dataset = Dataset(config)
logger.info(dataset)

model = BPRMF(config, dataset).to(config['device'])
logger.info(model)

# If you want to customize the evaluation setting,
# please refer to `data_preparation()` in `data/utils.py`.
train_data, test_data, valid_data = data_preparation(config, model, dataset)

trainer = Trainer(config, model)
# trainer.resume_checkpoint('saved/model_best.pth')
best_valid_score, _ = trainer.fit(train_data, valid_data)
result = trainer.evaluate(test_data)
logger.info(best_valid_score)
