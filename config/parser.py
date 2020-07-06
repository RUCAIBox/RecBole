from optparse import OptionParser


class Parser(object):
    def __init__(self):
        parser = OptionParser()
        parser.add_option('--gpu_id', dest='gpu_id', default=0, type='int', help='GPU id that be used')
        parser.add_option('--seed', dest='seed', default=2020, type='int', help='random seed')
        parser.add_option('--data.num_workers', dest='data.num_workers', default=0, type='int',
                          help='multi-process when load data')
        parser.add_option('--data.separator', dest='data.separator', default=0, type='int', help='data separator')
        parser.add_option('--process.remove_lower_value_by_key.key', dest='process.remove_lower_value_by_key.key',
                          default='rating', type='string', help='data filter')
        parser.add_option('--process.remove_lower_value_by_key.min_remain_value',
                          dest='process.remove_lower_value_by_key.min_remain_value',
                          default=3, type='int', help='data filter')
        parser.add_option('--process.neg_sample_to.num',
                          dest='process.neg_sample_to.num',
                          default=100, type='int', help='number of neg samples')
        parser.add_option('--eval.metric', dest='eval.metric', default='["Recall", "Hit", "MRR"]',
                          type='str', help='evaluation metric')
        parser.add_option('--eval.topk', dest='eval.topk', default='[10, 20]',
                          type='str', help='evaluation K')
        parser.add_option('--eval.candidate_neg', dest='eval.candidate_neg', default=0,
                          type='int', help='number of candidate neg items when testing')
        parser.add_option('--eval.test_batch_size', dest='eval.test_batch_size', default=128,
                          type='int', help='test batch size')
        parser.add_option('--model.learning_rate', dest='model.learning_rate', default=0.001,
                          type='float', help='learning rate')

        (self.options, self.args) = parser.parse_args()

    def getargs(self):
        return self.options.__dict__


if __name__ == '__main__':
    parser = Parser()
    args = parser.getargs()
    print(args)

