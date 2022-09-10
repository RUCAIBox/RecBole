# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [LightGCN](https://recbole.io/docs/user_guide/model/general/lightgcn.html)

- **Time cost**: 3119.76s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate in [5e-4,1e-3,2e-3] 
  n_layers in [1,2,3,4] 
  reg_weight in [1e-05,1e-04,1e-03,1e-02]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 5e-4  
  n_layers: 1  
  reg_weight: 0.01
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0005, n_layers:2, reg_weight:0.001
  Valid result:
  recall@10 : 0.1547    mrr@10 : 0.3612    ndcg@10 : 0.1964    hit@10 : 0.6897    precision@10 : 0.1458
  Test result:
  recall@10 : 0.1714    mrr@10 : 0.4196    ndcg@10 : 0.2376    hit@10 : 0.7079    precision@10 : 0.1769

  learning_rate:0.001, n_layers:1, reg_weight:0.0001
  Valid result:
  recall@10 : 0.1644    mrr@10 : 0.3648    ndcg@10 : 0.2043    hit@10 : 0.7081    precision@10 : 0.152
  Test result:
  recall@10 : 0.181     mrr@10 : 0.4302    ndcg@10 : 0.2466    hit@10 : 0.7302    precision@10 : 0.183

  learning_rate:0.001, n_layers:4, reg_weight:0.001
  Valid result:
  recall@10 : 0.1688    mrr@10 : 0.3653    ndcg@10 : 0.2064    hit@10 : 0.7157    precision@10 : 0.1533
  Test result:
  recall@10 : 0.1857    mrr@10 : 0.4303    ndcg@10 : 0.2489    hit@10 : 0.735     precision@10 : 0.1846
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 48/48 [41:35:48<00:00, 3119.76s/trial, best loss: -0.2065]
  best params:  {'learning_rate': 0.0005, 'n_layers': 1, 'reg_weight': 0.01}
  best result: 
  {'model': 'LightGCN', 'best_valid_score': 0.2065, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1659), ('mrr@10', 0.3687), ('ndcg@10', 0.2065), ('hit@10', 0.7098), ('precision@10', 0.1527)]), 'test_result': OrderedDict([('recall@10', 0.1823), ('mrr@10', 0.4312), ('ndcg@10', 0.2476), ('hit@10', 0.7316), ('precision@10', 0.1835)])}
  ```
