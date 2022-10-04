# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [ENMF](https://recbole.io/docs/user_guide/model/general/enmf.html)

- **Time cost**: 162.08s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate in [0.005,0.01,0.05]
  dropout_prob in [0.3,0.5,0.7]
  negative_weight in [0.1,0.2,0.5]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.005  
  dropout_prob: 0.3  
  negative_weight: 0.5
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  dropout_prob:0.3, learning_rate:0.005, negative_weight:0.5
  Valid result:
  recall@10 : 0.1816    mrr@10 : 0.3886    ndcg@10 : 0.2197    hit@10 : 0.7384    precision@10 : 0.1587
  Test result:
  recall@10 : 0.2007    mrr@10 : 0.4521    ndcg@10 : 0.2629    hit@10 : 0.7605    precision@10 : 0.1906

  dropout_prob:0.5, learning_rate:0.005, negative_weight:0.1
  Valid result:
  recall@10 : 0.1712    mrr@10 : 0.3552    ndcg@10 : 0.2014    hit@10 : 0.7184    precision@10 : 0.1487
  Test result:
  recall@10 : 0.1871    mrr@10 : 0.4127    ndcg@10 : 0.239     hit@10 : 0.7419    precision@10 : 0.1765

  dropout_prob:0.7, learning_rate:0.005, negative_weight:0.2
  Valid result:
  recall@10 : 0.1688    mrr@10 : 0.3647    ndcg@10 : 0.2048    hit@10 : 0.7149    precision@10 : 0.1509
  Test result:
  recall@10 : 0.1847    mrr@10 : 0.4302    ndcg@10 : 0.2471    hit@10 : 0.7343    precision@10 : 0.1815
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 27/27 [1:12:56<00:00, 162.08s/trial, best loss: -0.2197]
  best params:  {'dropout_prob': 0.3, 'learning_rate': 0.005, 'negative_weight': 0.5}
  best result: 
  {'model': 'ENMF', 'best_valid_score': 0.2197, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1816), ('mrr@10', 0.3886), ('ndcg@10', 0.2197), ('hit@10', 0.7384), ('precision@10', 0.1587)]), 'test_result': OrderedDict([('recall@10', 0.2007), ('mrr@10', 0.4521), ('ndcg@10', 0.2629), ('hit@10', 0.7605), ('precision@10', 0.1906)])}
  ```
