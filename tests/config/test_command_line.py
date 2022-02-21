# @Time   : 2020/10/19
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2021/7/1
# @Author : Xingyu Pan
# @Email  : xy_pan@foxmail.com


from recbole.config import Config


if __name__ == '__main__':

    config = Config(model='BPR', dataset='ml-100k')

    # command line
    assert config['use_gpu'] is False
    assert config['valid_metric'] == 'Recall@10'
    assert config['metrics'] == ['Recall']     # bug

    # priority
    assert config['epochs'] == 200
    assert config['learning_rate'] == 0.3

    print('------------------------------------------------------------')
    print('OK')
