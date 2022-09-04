# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [NARM](https://recbole.io/docs/user_guide/model/sequential/narm.html)

- **Time cost**: 1776.89s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.005,0.001,0.0005,0.0001]
  num_layers choice [1,2]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.005
  num_layers: 1
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.005, num_layers:1
  Valid result:
  recall@10 : 0.2843    mrr@10 : 0.115    ndcg@10 : 0.1544    hit@10 : 0.2843    precision@10 : 0.0284
  Test result:
  recall@10 : 0.2662    mrr@10 : 0.1108    ndcg@10 : 0.1471    hit@10 : 0.2662    precision@10 : 0.0266

  learning_rate:0.0001, num_layers:1
  Valid result:
  recall@10 : 0.2251    mrr@10 : 0.0844    ndcg@10 : 0.117    hit@10 : 0.2251    precision@10 : 0.0225
  Test result:
  recall@10 : 0.2228    mrr@10 : 0.0885    ndcg@10 : 0.1198    hit@10 : 0.2228    precision@10 : 0.0223

  learning_rate:0.0005, num_layers:1
  Valid result:
  recall@10 : 0.2697    mrr@10 : 0.1084    ndcg@10 : 0.146    hit@10 : 0.2697    precision@10 : 0.027
  Test result:
  recall@10 : 0.2606    mrr@10 : 0.1027    ndcg@10 : 0.1394    hit@10 : 0.2606    precision@10 : 0.0261

  learning_rate:0.001, num_layers:1
  Valid result:
  recall@10 : 0.2788    mrr@10 : 0.1099    ndcg@10 : 0.1491    hit@10 : 0.2788    precision@10 : 0.0279
  Test result:
  recall@10 : 0.2617    mrr@10 : 0.1081    ndcg@10 : 0.1438    hit@10 : 0.2617    precision@10 : 0.0262

  learning_rate:0.0001, num_layers:2
  Valid result:
  recall@10 : 0.2251    mrr@10 : 0.0844    ndcg@10 : 0.117    hit@10 : 0.2251    precision@10 : 0.0225
  Test result:
  recall@10 : 0.2228    mrr@10 : 0.0885    ndcg@10 : 0.1198    hit@10 : 0.2228    precision@10 : 0.0223

  learning_rate:0.0005, num_layers:2
  Valid result:
  recall@10 : 0.2697    mrr@10 : 0.1084    ndcg@10 : 0.146    hit@10 : 0.2697    precision@10 : 0.027
  Test result:
  recall@10 : 0.2606    mrr@10 : 0.1027    ndcg@10 : 0.1394    hit@10 : 0.2606    precision@10 : 0.0261

  learning_rate:0.001, num_layers:2
  Valid result:
  recall@10 : 0.2788    mrr@10 : 0.1099    ndcg@10 : 0.1491    hit@10 : 0.2788    precision@10 : 0.0279
  Test result:
  recall@10 : 0.2617    mrr@10 : 0.1081    ndcg@10 : 0.1438    hit@10 : 0.2617    precision@10 : 0.0262

  learning_rate:0.005, num_layers:2
  Valid result:
  recall@10 : 0.2843    mrr@10 : 0.115    ndcg@10 : 0.1544    hit@10 : 0.2843    precision@10 : 0.0284
  Test result:
  recall@10 : 0.2662    mrr@10 : 0.1108    ndcg@10 : 0.1471    hit@10 : 0.2662    precision@10 : 0.0266
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 8/8 [3:56:55<00:00, 1776.89s/trial, best loss: -0.1544]
  best params:  {'learning_rate': 0.005, 'num_layers': 1}
  best result: 
  {'model': 'NARM', 'best_valid_score': 0.1544, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2843), ('mrr@10', 0.115), ('ndcg@10', 0.1544), ('hit@10', 0.2843), ('precision@10', 0.0284)]), 'test_result': OrderedDict([('recall@10', 0.2662), ('mrr@10', 0.1108), ('ndcg@10', 0.1471), ('hit@10', 0.2662), ('precision@10', 0.0266)])}
  ```
