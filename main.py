import json
from argparse import ArgumentParser
from run_test import whole_process

parser = ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='BPRMF', help='name of models')
parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
parser.add_argument('--epochs', '-e', type=int, default=1, help='num of running epochs')

args = parser.parse_args()

args_dict = {
    'model': args.model,
    'dataset': args.dataset,
    'epochs': args.epochs
}

with open('presets.json', 'r', encoding='utf-8') as preset_file:
    presets_dict = json.load(preset_file)

token = '-'.join([args.model, args.dataset])
if token in presets_dict:
    print('Hit preset: [{}]'.format(token))
    args_dict.update(presets_dict)

whole_process(config_dict=args_dict)
