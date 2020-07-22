from .dataloader import *
from config import EvalSetting

def data_preparation(config, model, dataset):
    es = EvalSetting(config)

    # es.group_by_user()
    es.shuffle()
    es.split_by_ratio(config['split_ratio'])

    train_dataset, test_dataset, valid_dataset = dataset.build(es)

    es.neg_sample_by(1)
    train_data = dataloader_construct(
        name='train',
        config=config,
        eval_setting=es,
        dataset=train_dataset,
        dl_type=model.type,
        dl_format=config['input_format'],
        batch_size=config['train_batch_size'],
        shuffle=True
    )

    es.neg_sample_to(config['test_neg_sample_num'])
    test_data, valid_data = dataloader_construct(
        name='evaluation',
        config=config,
        eval_setting=es,
        dataset=[test_dataset, valid_dataset],
        dl_type=model.type,
        batch_size=config['eval_batch_size']
    )

    return train_data, test_data, valid_data

def dataloader_construct(name, config, eval_setting, dataset,
                         dl_type='general', dl_format='pointwise',
                         batch_size=1, shuffle=False):
    if not isinstance(dataset, list):
        dataset = [dataset]

    if not isinstance(batch_size, list):
        batch_size = [batch_size] * len(dataset)

    if len(dataset) != len(batch_size):
        raise ValueError('dataset {} and batch_size {} should have the same length'.format(dataset, batch_size))

    print('Build [{}] DataLoader for [{}] with format [{}]\n'.format(dl_type, name, dl_format))
    print(eval_setting)
    print('batch_size = {}, shuffle = {}\n'.format(batch_size, shuffle))

    if dl_type == 'general':
        DataLoader = GeneralDataLoader
    else:
        raise NotImplementedError('dl_type [{}] has not been implemented'.format(dl_type))

    ret = []

    for i, ds in enumerate(dataset):
        dl = DataLoader(
            config=config,
            dataset=ds,
            neg_sample_args=eval_setting.neg_sample_args,
            batch_size=batch_size[i],
            dl_format=dl_format,
            shuffle=shuffle
        )
        ret.append(dl)
    
    if len(ret) == 1:
        return ret[0]
    else:
        return ret

