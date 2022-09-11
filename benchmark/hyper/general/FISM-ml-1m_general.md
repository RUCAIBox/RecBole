# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [FISM](https://recbole.io/docs/user_guide/model/general/fism.html)

- **Time cost**: 4666.78s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [5e-4,1e-4,5e-3,1e-3] 
  embedding_size choice [64] 
  regs choice ['[1e-7, 1e-7]', '[0, 0]'] 
  alpha choice [0.0]
  ```

- **Best parameters**:

  ```yaml
  alpha: 0.0  
  embedding_size: 64  
  learning_rate: 1e-3  
  regs: [0,0]
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  alpha:0.0, embedding_size:64, learning_rate:0.0001, regs:'[1e-7,1e-7]'
  Valid result:
  recall@10 : 0.0748    mrr@10 : 0.2081    ndcg@10 : 0.1038    hit@10 : 0.4648    precision@10 : 0.0822
  Test result:
  recall@10 : 0.0775    mrr@10 : 0.2338    ndcg@10 : 0.1182    hit@10 : 0.4779    precision@10 : 0.0938

  alpha:0.0, embedding_size:64, learning_rate:0.001, regs:'[0,0]'
  Valid result:
  recall@10 : 0.0752    mrr@10 : 0.212     ndcg@10 : 0.1053    hit@10 : 0.4606    precision@10 : 0.0826
  Test result:
  recall@10 : 0.0781    mrr@10 : 0.2352    ndcg@10 : 0.119     hit@10 : 0.4767    precision@10 : 0.0938

  alpha:0.0, embedding_size:64, learning_rate:0.005, regs:'[0,0]'
  Valid result:
  recall@10 : 0.0743    mrr@10 : 0.2132    ndcg@10 : 0.104     hit@10 : 0.4591    precision@10 : 0.0817
  Test result:
  recall@10 : 0.0783    mrr@10 : 0.2393    ndcg@10 : 0.1189    hit@10 : 0.4789    precision@10 : 0.0939
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 8/8 [[10:22:14<00:00, 4666.78s/trial, best loss: -0.1053]
  best params:  {'alpha': 0.0, 'embedding_size': 64, 'learning_rate': 0.001, 'regs': '[0,0]'}
  best result:
  {'model': 'FISM', 'best_valid_score': 0.1053, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0752), ('mrr@10', 0.212), ('ndcg@10', 0.1053), ('hit@10', 0.4606), ('precision@10', 0.0826)]), 'test_result': OrderedDict([('recall@10', 0.0781), ('mrr@10', 0.2352), ('ndcg@10', 0.119), ('hit@10', 0.4767), ('precision@10', 0.0938)])}
  ```
