import yaml
from argparse import ArgumentParser
from run_test import whole_process

parser = ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='BPRMF', help='name of models')
parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
parser.add_argument('--epochs', '-e', type=int, default=1, help='num of running epochs')

args, _ = parser.parse_known_args()

args_dict = {
    'model': args.model,
    'dataset': args.dataset,
    'epochs': args.epochs
}

with open('presets.yaml', 'r', encoding='utf-8') as preset_file:
    presets_dict = yaml.load(preset_file.read(), Loader=yaml.FullLoader)

token = '-'.join([args.model, args.dataset])
if token in presets_dict:
    print('Hit preset: [{}]'.format(token))
    args_dict.update(presets_dict[token])

whole_process(config_dict=args_dict)
