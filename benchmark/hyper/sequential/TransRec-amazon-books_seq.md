# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [TransRec](https://recbole.io/docs/user_guide/model/sequential/transrec.html)

- **Time cost**: 17555.58s/trial

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
  recall@10 : 0.1615    mrr@10 : 0.0462    ndcg@10 : 0.0733    hit@10 : 0.1615    precision@10 : 0.0162
  Test result:
  recall@10 : 0.1194    mrr@10 : 0.0335    ndcg@10 : 0.0536    hit@10 : 0.1194    precision@10 : 0.0119

  learning_rate:0.005
  Valid result:
  recall@10 : 0.1555    mrr@10 : 0.0443    ndcg@10 : 0.0704    hit@10 : 0.1555    precision@10 : 0.0155
  Test result:
  recall@10 : 0.1165    mrr@10 : 0.0325    ndcg@10 : 0.0522    hit@10 : 0.1165    precision@10 : 0.0116

  learning_rate:0.001
  Valid result:
  recall@10 : 0.1555    mrr@10 : 0.045    ndcg@10 : 0.071    hit@10 : 0.1555    precision@10 : 0.0156
  Test result:
  recall@10 : 0.116    mrr@10 : 0.0325    ndcg@10 : 0.052    hit@10 : 0.116    precision@10 : 0.0116

  learning_rate:0.003
  Valid result:
  recall@10 : 0.1573    mrr@10 : 0.0452    ndcg@10 : 0.0716    hit@10 : 0.1573    precision@10 : 0.0157
  Test result:
  recall@10 : 0.1176    mrr@10 : 0.0329    ndcg@10 : 0.0527    hit@10 : 0.1176    precision@10 : 0.0118

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.1538    mrr@10 : 0.044    ndcg@10 : 0.0698    hit@10 : 0.1538    precision@10 : 0.0154
  Test result:
  recall@10 : 0.1129    mrr@10 : 0.0322    ndcg@10 : 0.0511    hit@10 : 0.1129    precision@10 : 0.0113
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [24:22:57<00:00, 17555.58s/trial, best loss: -0.0733]
  best params:  {'learning_rate': 0.0005}
  best result: 
  {'model': 'TransRec', 'best_valid_score': 0.0733, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1615), ('mrr@10', 0.0462), ('ndcg@10', 0.0733), ('hit@10', 0.1615), ('precision@10', 0.0162)]), 'test_result': OrderedDict([('recall@10', 0.1194), ('mrr@10', 0.0335), ('ndcg@10', 0.0536), ('hit@10', 0.1194), ('precision@10', 0.0119)])}
  ```
