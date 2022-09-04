# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [Caser](https://recbole.io/docs/user_guide/model/sequential/caser.html)

- **Time cost**: 43720.39s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0003, 0.0005, 0.001, 0.003, 0.005]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0003
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0005
  Valid result:
  recall@10 : 0.1774    mrr@10 : 0.0759    ndcg@10 : 0.0997    hit@10 : 0.1774    precision@10 : 0.0177
  Test result:
  recall@10 : 0.1192    mrr@10 : 0.0513    ndcg@10 : 0.0671    hit@10 : 0.1192    precision@10 : 0.0119

  learning_rate:0.003
  Valid result:
  recall@10 : 0.1586    mrr@10 : 0.0653    ndcg@10 : 0.0871    hit@10 : 0.1586    precision@10 : 0.0159
  Test result:
  recall@10 : 0.1028    mrr@10 : 0.0423    ndcg@10 : 0.0564    hit@10 : 0.1028    precision@10 : 0.0103

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.1799    mrr@10 : 0.0794    ndcg@10 : 0.103    hit@10 : 0.1799    precision@10 : 0.018
  Test result:
  recall@10 : 0.1258    mrr@10 : 0.0533    ndcg@10 : 0.0702    hit@10 : 0.1258    precision@10 : 0.0126

  learning_rate:0.001
  Valid result:
  recall@10 : 0.1746    mrr@10 : 0.0733    ndcg@10 : 0.097    hit@10 : 0.1746    precision@10 : 0.0175
  Test result:
  recall@10 : 0.1203    mrr@10 : 0.0519    ndcg@10 : 0.0678    hit@10 : 0.1203    precision@10 : 0.012

  learning_rate:0.005
  Valid result:
  recall@10 : 0.16    mrr@10 : 0.0656    ndcg@10 : 0.0877    hit@10 : 0.16    precision@10 : 0.016
  Test result:
  recall@10 : 0.1025    mrr@10 : 0.0407    ndcg@10 : 0.0551    hit@10 : 0.1025    precision@10 : 0.0102
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [60:43:21<00:00, 43720.39s/trial, best loss: -0.103]
  best params:  {'learning_rate': 0.0003}
  best result: 
  {'model': 'Caser', 'best_valid_score': 0.103, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1799), ('mrr@10', 0.0794), ('ndcg@10', 0.103), ('hit@10', 0.1799), ('precision@10', 0.018)]), 'test_result': OrderedDict([('recall@10', 0.1258), ('mrr@10', 0.0533), ('ndcg@10', 0.0702), ('hit@10', 0.1258), ('precision@10', 0.0126)])}
  ```
