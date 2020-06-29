from config import Config
from dataset import ML100kDataset
from model.general_recommender.bprmf import BPRMF
from trainer import Trainer


config = Config('properties/overall.config')
config.init()

dataset = ML100kDataset(config)
train_data, test_data = dataset.preprocessing(
    workflow=['split']
)

model = BPRMF(config, dataset)
trainer = Trainer(config, model)
trainer.resume_checkpoint('save/model_best.pth')
trainer.train(train_data)
test_result = trainer.test(test_data)
print(test_result)
trainer.plot_train_loss(show=True)
