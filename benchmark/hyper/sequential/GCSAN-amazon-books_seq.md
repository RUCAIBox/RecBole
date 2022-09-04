# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [GCSAN](https://recbole.io/docs/user_guide/model/sequential/gcsan.html)

- **Time cost**: 20405.22s/trial

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
  recall@10 : 0.2083    mrr@10 : 0.1108    ndcg@10 : 0.1337    hit@10 : 0.2083    precision@10 : 0.0208
  Test result:
  recall@10 : 0.1622    mrr@10 : 0.0843    ndcg@10 : 0.1026    hit@10 : 0.1622    precision@10 : 0.0162

  learning_rate:0.003
  Valid result:
  recall@10 : 0.2115    mrr@10 : 0.1076    ndcg@10 : 0.1321    hit@10 : 0.2115    precision@10 : 0.0212
  Test result:
  recall@10 : 0.1667    mrr@10 : 0.0835    ndcg@10 : 0.103    hit@10 : 0.1667    precision@10 : 0.0167

  learning_rate:0.005
  Valid result:
  recall@10 : 0.2065    mrr@10 : 0.102    ndcg@10 : 0.1266    hit@10 : 0.2065    precision@10 : 0.0206
  Test result:
  recall@10 : 0.1618    mrr@10 : 0.0787    ndcg@10 : 0.0981    hit@10 : 0.1618    precision@10 : 0.0162

  learning_rate:0.001
  Valid result:
  recall@10 : 0.2151    mrr@10 : 0.1101    ndcg@10 : 0.1348    hit@10 : 0.2151    precision@10 : 0.0215
  Test result:
  recall@10 : 0.1706    mrr@10 : 0.0848    ndcg@10 : 0.105    hit@10 : 0.1706    precision@10 : 0.0171

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.2079    mrr@10 : 0.1089    ndcg@10 : 0.1322    hit@10 : 0.2079    precision@10 : 0.0208
  Test result:
  recall@10 : 0.1644    mrr@10 : 0.0828    ndcg@10 : 0.1019    hit@10 : 0.1644    precision@10 : 0.0164
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [28:20:26<00:00, 20405.22s/trial, best loss: -0.1348]
  best params:  {'learning_rate': 0.001}
  best result: 
  {'model': 'GCSAN', 'best_valid_score': 0.1348, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2151), ('mrr@10', 0.1101), ('ndcg@10', 0.1348), ('hit@10', 0.2151), ('precision@10', 0.0215)]), 'test_result': OrderedDict([('recall@10', 0.1706), ('mrr@10', 0.0848), ('ndcg@10', 0.105), ('hit@10', 0.1706), ('precision@10', 0.0171)])}
  ```
