from config import Config
from data import ML100kDataset
from model.general_recommender.bprmf import BPRMF
from trainer import Trainer
from utils import Logger

config = Config('properties/overall.config')
config.init()

logger = Logger('0630.log')

dataset = ML100kDataset(config)

train_data, test_data, valid_data = dataset.preprocessing(
    workflow={
        'preprocessing': ['remove_lower_value_by_key', 'split_by_ratio'],
        'train': ['neg_sample_1by1'],
        'test': ['neg_sample_to'],
        'valid': ['neg_sample_to']
    }
)

model = BPRMF(config, dataset).to(config['device'])
trainer = Trainer(config, logger, model)
# trainer.resume_checkpoint('save/model_best.pth')
trainer.train(train_data, valid_data)
test_result = trainer.test(test_data)
print(test_result)
trainer.plot_train_loss(show=True)
