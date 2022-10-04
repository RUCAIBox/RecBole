# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [CORE](https://recbole.io/docs/user_guide/model/sequential/core.html)

- **Time cost**: 1645.05s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate in [0.0005, 0.001, 0.003]
  temperature in [0.05, 0.07, 0.1, 0.2]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0005
  temperature: 0.07
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0005, temperature:0.07
  Valid result:
  recall@10 : 0.1527    mrr@10 : 0.0375    ndcg@10 : 0.0638    hit@10 : 0.1527    precision@10 : 0.0153
  Test result:
  recall@10 : 0.1601    mrr@10 : 0.0403    ndcg@10 : 0.0677    hit@10 : 0.1601    precision@10 : 0.016

  learning_rate:0.003, temperature:0.05
  Valid result:
  recall@10 : 0.1472    mrr@10 : 0.0373    ndcg@10 : 0.0624    hit@10 : 0.1472    precision@10 : 0.0147
  Test result:
  recall@10 : 0.151    mrr@10 : 0.0385    ndcg@10 : 0.0642    hit@10 : 0.151    precision@10 : 0.0151

  learning_rate:0.003, temperature:0.07
  Valid result:
  recall@10 : 0.1505    mrr@10 : 0.0364    ndcg@10 : 0.0624    hit@10 : 0.1505    precision@10 : 0.0151
  Test result:
  recall@10 : 0.1573    mrr@10 : 0.0404    ndcg@10 : 0.0671    hit@10 : 0.1573    precision@10 : 0.0157

  learning_rate:0.003, temperature:0.1
  Valid result:
  recall@10 : 0.1543    mrr@10 : 0.037    ndcg@10 : 0.0638    hit@10 : 0.1543    precision@10 : 0.0154
  Test result:
  recall@10 : 0.1596    mrr@10 : 0.0394    ndcg@10 : 0.0669    hit@10 : 0.1596    precision@10 : 0.016

  learning_rate:0.001, temperature:0.07
  Valid result:
  recall@10 : 0.1518    mrr@10 : 0.0365    ndcg@10 : 0.0627    hit@10 : 0.1518    precision@10 : 0.0152
  Test result:
  recall@10 : 0.157    mrr@10 : 0.0396    ndcg@10 : 0.0665    hit@10 : 0.157    precision@10 : 0.0157

  learning_rate:0.0005, temperature:0.1
  Valid result:
  recall@10 : 0.1505    mrr@10 : 0.0373    ndcg@10 : 0.0632    hit@10 : 0.1505    precision@10 : 0.0151
  Test result:
  recall@10 : 0.1598    mrr@10 : 0.0394    ndcg@10 : 0.0669    hit@10 : 0.1598    precision@10 : 0.016

  learning_rate:0.0005, temperature:0.05
  Valid result:
  recall@10 : 0.1535    mrr@10 : 0.0369    ndcg@10 : 0.0635    hit@10 : 0.1535    precision@10 : 0.0153
  Test result:
  recall@10 : 0.1551    mrr@10 : 0.0386    ndcg@10 : 0.0652    hit@10 : 0.1551    precision@10 : 0.0155

  learning_rate:0.0005, temperature:0.2
  Valid result:
  recall@10 : 0.1276    mrr@10 : 0.0362    ndcg@10 : 0.0572    hit@10 : 0.1276    precision@10 : 0.0128
  Test result:
  recall@10 : 0.1319    mrr@10 : 0.0373    ndcg@10 : 0.059    hit@10 : 0.1319    precision@10 : 0.0132

  learning_rate:0.001, temperature:0.05
  Valid result:
  recall@10 : 0.1472    mrr@10 : 0.0366    ndcg@10 : 0.0618    hit@10 : 0.1472    precision@10 : 0.0147
  Test result:
  recall@10 : 0.1537    mrr@10 : 0.0395    ndcg@10 : 0.0656    hit@10 : 0.1537    precision@10 : 0.0154

  learning_rate:0.003, temperature:0.2
  Valid result:
  recall@10 : 0.1273    mrr@10 : 0.0375    ndcg@10 : 0.0581    hit@10 : 0.1273    precision@10 : 0.0127
  Test result:
  recall@10 : 0.1268    mrr@10 : 0.0357    ndcg@10 : 0.0567    hit@10 : 0.1268    precision@10 : 0.0127
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 12/12 [5:29:06<00:00, 1645.05s/trial, best loss: -0.0638]
  best params:  {'learning_rate': 0.0005, 'temperature': 0.07}
  best result: 
  {'model': 'CORE', 'best_valid_score': 0.0638, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1527), ('mrr@10', 0.0375), ('ndcg@10', 0.0638), ('hit@10', 0.1527), ('precision@10', 0.0153)]), 'test_result': OrderedDict([('recall@10', 0.1601), ('mrr@10', 0.0403), ('ndcg@10', 0.0677), ('hit@10', 0.1601), ('precision@10', 0.016)])}
  ```
