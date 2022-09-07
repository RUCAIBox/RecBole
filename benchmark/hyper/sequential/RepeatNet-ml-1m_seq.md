# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [RepeatNet](https://recbole.io/docs/user_guide/model/sequential/repeatnet.html)

- **Time cost**: 21803.70s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0003, 0.0005, 0.001, 0.003, 0.005]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0005
  Valid result:
  recall@10 : 0.3042    mrr@10 : 0.1392    ndcg@10 : 0.1779    hit@10 : 0.3042    precision@10 : 0.0304
  Test result:
  recall@10 : 0.2808    mrr@10 : 0.1293    ndcg@10 : 0.1649    hit@10 : 0.2808    precision@10 : 0.0281

  learning_rate:0.003
  Valid result:
  recall@10 : 0.3037    mrr@10 : 0.1383    ndcg@10 : 0.177    hit@10 : 0.3037    precision@10 : 0.0304
  Test result:
  recall@10 : 0.2871    mrr@10 : 0.1323    ndcg@10 : 0.1684    hit@10 : 0.2871    precision@10 : 0.0287

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.3048    mrr@10 : 0.1406    ndcg@10 : 0.1791    hit@10 : 0.3048    precision@10 : 0.0305
  Test result:
  recall@10 : 0.2833    mrr@10 : 0.1305    ndcg@10 : 0.1663    hit@10 : 0.2833    precision@10 : 0.0283

  learning_rate:0.001
  Valid result:
  recall@10 : 0.3071    mrr@10 : 0.1401    ndcg@10 : 0.1792    hit@10 : 0.3071    precision@10 : 0.0307
  Test result:
  recall@10 : 0.2844    mrr@10 : 0.1301    ndcg@10 : 0.1662    hit@10 : 0.2844    precision@10 : 0.0284

  learning_rate:0.005
  Valid result:
  recall@10 : 0.3053    mrr@10 : 0.1366    ndcg@10 : 0.1762    hit@10 : 0.3053    precision@10 : 0.0305
  Test result:
  recall@10 : 0.2823    mrr@10 : 0.1302    ndcg@10 : 0.1658    hit@10 : 0.2823    precision@10 : 0.0282
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [30:16:58<00:00, 21803.70s/trial, best loss: -0.1792]
  best params:  {'learning_rate': 0.001}
  best result: 
  {'model': 'RepeatNet', 'best_valid_score': 0.1792, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.3071), ('mrr@10', 0.1401), ('ndcg@10', 0.1792), ('hit@10', 0.3071), ('precision@10', 0.0307)]), 'test_result': OrderedDict([('recall@10', 0.2844), ('mrr@10', 0.1301), ('ndcg@10', 0.1662), ('hit@10', 0.2844), ('precision@10', 0.0284)])}
  ```
