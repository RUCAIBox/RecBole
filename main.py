from config import Config
from data import Dataset
from model.general_recommender.bprmf import BPRMF
from trainer import Trainer
from utils import Logger

config = Config('properties/overall.config')
config.init()

logger = Logger(config)

dataset = Dataset(config)
print(dataset)

train_data, test_data, valid_data = dataset.build(
    inter_filter_lowest_val=config['lowest_val'],
    inter_filter_highest_val=config['highest_val'],
    split_by_ratio=[config['train_split_ratio'], config['valid_split_ratio'], config['test_split_ratio']],
    train_batch_size=config['train_batch_size'],
    test_batch_size=config['test_batch_size'],
    valid_batch_size=config['valid_batch_size'],
    pairwise=True,
    neg_sample_by=1,
    neg_sample_to=config['test_neg_sample_num']
)
model = BPRMF(config, dataset).to(config['device'])
print(model)

trainer = Trainer(config, model, logger)
# trainer.resume_checkpoint('saved/model_best.pth')
best_valid_score, _ = trainer.fit(train_data, valid_data)
result = trainer.evaluate(test_data)
print(best_valid_score)
