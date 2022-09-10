# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [SINE](https://recbole.io/docs/user_guide/model/sequential/sine.html)

- **Time cost**: 12833.78s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0003, 0.0005, 0.001, 0.003, 0.005]
  ```

- **Best parameters**:

  ```
  learning_rate: 0.0005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0005
  Valid result:
  recall@10 : 0.1703    mrr@10 : 0.0578    ndcg@10 : 0.084    hit@10 : 0.1703    precision@10 : 0.017
  Test result:
  recall@10 : 0.1336    mrr@10 : 0.0443    ndcg@10 : 0.0651    hit@10 : 0.1336    precision@10 : 0.0134

  learning_rate:0.005
  Valid result:
  recall@10 : 0.1687    mrr@10 : 0.0515    ndcg@10 : 0.0787    hit@10 : 0.1687    precision@10 : 0.0169
  Test result:
  recall@10 : 0.1266    mrr@10 : 0.0378    ndcg@10 : 0.0582    hit@10 : 0.1266    precision@10 : 0.0127

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.1563    mrr@10 : 0.0543    ndcg@10 : 0.078    hit@10 : 0.1563    precision@10 : 0.0156
  Test result:
  recall@10 : 0.118    mrr@10 : 0.0402    ndcg@10 : 0.0583    hit@10 : 0.118    precision@10 : 0.0118

  learning_rate:0.001
  Valid result:
  recall@10 : 0.1752    mrr@10 : 0.0563    ndcg@10 : 0.0839    hit@10 : 0.1752    precision@10 : 0.0175
  Test result:
  recall@10 : 0.1338    mrr@10 : 0.0427    ndcg@10 : 0.0638    hit@10 : 0.1338    precision@10 : 0.0134

  learning_rate:0.003
  Valid result:
  recall@10 : 0.166    mrr@10 : 0.0511    ndcg@10 : 0.0777    hit@10 : 0.166    precision@10 : 0.0166
  Test result:
  recall@10 : 0.1253    mrr@10 : 0.0377    ndcg@10 : 0.0579    hit@10 : 0.1253    precision@10 : 0.0125
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [17:49:28<00:00, 12833.78s/trial, best loss: -0.084]
  best params:  {'learning_rate': 0.0005}
  best result: 
  {'model': 'SINE', 'best_valid_score': 0.084, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1703), ('mrr@10', 0.0578), ('ndcg@10', 0.084), ('hit@10', 0.1703), ('precision@10', 0.017)]), 'test_result': OrderedDict([('recall@10', 0.1336), ('mrr@10', 0.0443), ('ndcg@10', 0.0651), ('hit@10', 0.1336), ('precision@10', 0.0134)])}
  ```
