# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [STAMP](https://recbole.io/docs/user_guide/model/sequential/stamp.html)

- **Time cost**: 2954.26s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0003, 0.0005, 0.001, 0.003, 0.005]
  ```

- **Best parameters**:

  ```
  learning_rate: 0.003
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.001
  Valid result:
  recall@10 : 0.1848    mrr@10 : 0.0999    ndcg@10 : 0.1199    hit@10 : 0.1848    precision@10 : 0.0185
  Test result:
  recall@10 : 0.1465    mrr@10 : 0.0758    ndcg@10 : 0.0924    hit@10 : 0.1465    precision@10 : 0.0147

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.1791    mrr@10 : 0.0999    ndcg@10 : 0.1185    hit@10 : 0.1791    precision@10 : 0.0179
  Test result:
  recall@10 : 0.1432    mrr@10 : 0.0773    ndcg@10 : 0.0927    hit@10 : 0.1432    precision@10 : 0.0143

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.1772    mrr@10 : 0.0979    ndcg@10 : 0.1165    hit@10 : 0.1772    precision@10 : 0.0177
  Test result:
  recall@10 : 0.138    mrr@10 : 0.0748    ndcg@10 : 0.0896    hit@10 : 0.138    precision@10 : 0.0138

  learning_rate:0.003
  Valid result:
  recall@10 : 0.1905    mrr@10 : 0.1048    ndcg@10 : 0.1249    hit@10 : 0.1905    precision@10 : 0.0191
  Test result:
  recall@10 : 0.1474    mrr@10 : 0.0792    ndcg@10 : 0.0952    hit@10 : 0.1474    precision@10 : 0.0147

  learning_rate:0.005
  Valid result:
  recall@10 : 0.173    mrr@10 : 0.1022    ndcg@10 : 0.1189    hit@10 : 0.173    precision@10 : 0.0173
  Test result:
  recall@10 : 0.1382    mrr@10 : 0.0785    ndcg@10 : 0.0925    hit@10 : 0.1382    precision@10 : 0.0138
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [4:06:11<00:00, 2954.26s/trial, best loss: -0.1249]
  best params:  {'learning_rate': 0.003}
  best result: 
  {'model': 'STAMP', 'best_valid_score': 0.1249, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1905), ('mrr@10', 0.1048), ('ndcg@10', 0.1249), ('hit@10', 0.1905), ('precision@10', 0.0191)]), 'test_result': OrderedDict([('recall@10', 0.1474), ('mrr@10', 0.0792), ('ndcg@10', 0.0952), ('hit@10', 0.1474), ('precision@10', 0.0147)])}
  ```
