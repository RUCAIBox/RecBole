# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [STAMP](https://recbole.io/docs/user_guide/model/sequential/stamp.html)

- **Time cost**: 2870.51s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0003, 0.0005, 0.001, 0.003, 0.005]
  ```

- **Best parameters**:

  ```
  learning_rate: 0.005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0005
  Valid result:
  recall@10 : 0.2258    mrr@10 : 0.0954    ndcg@10 : 0.1256    hit@10 : 0.2258    precision@10 : 0.0226
  Test result:
  recall@10 : 0.2107    mrr@10 : 0.0892    ndcg@10 : 0.1176    hit@10 : 0.2107    precision@10 : 0.0211

  learning_rate:0.005
  Valid result:
  recall@10 : 0.2331    mrr@10 : 0.1005    ndcg@10 : 0.1314    hit@10 : 0.2331    precision@10 : 0.0233
  Test result:
  recall@10 : 0.2147    mrr@10 : 0.0931    ndcg@10 : 0.1214    hit@10 : 0.2147    precision@10 : 0.0215

  learning_rate:0.001
  Valid result:
  recall@10 : 0.2271    mrr@10 : 0.0977    ndcg@10 : 0.1279    hit@10 : 0.2271    precision@10 : 0.0227
  Test result:
  recall@10 : 0.2223    mrr@10 : 0.0959    ndcg@10 : 0.1253    hit@10 : 0.2223    precision@10 : 0.0222

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.2266    mrr@10 : 0.0958    ndcg@10 : 0.1263    hit@10 : 0.2266    precision@10 : 0.0227
  Test result:
  recall@10 : 0.2065    mrr@10 : 0.0874    ndcg@10 : 0.1152    hit@10 : 0.2065    precision@10 : 0.0207

  learning_rate:0.003
  Valid result:
  recall@10 : 0.2276    mrr@10 : 0.0974    ndcg@10 : 0.1277    hit@10 : 0.2276    precision@10 : 0.0228
  Test result:
  recall@10 : 0.2188    mrr@10 : 0.0937    ndcg@10 : 0.1229    hit@10 : 0.2188    precision@10 : 0.0219
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [3:59:12<00:00, 2870.51s/trial, best loss: -0.1314]
  best params:  {'learning_rate': 0.005}
  best result: 
  {'model': 'STAMP', 'best_valid_score': 0.1314, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2331), ('mrr@10', 0.1005), ('ndcg@10', 0.1314), ('hit@10', 0.2331), ('precision@10', 0.0233)]), 'test_result': OrderedDict([('recall@10', 0.2147), ('mrr@10', 0.0931), ('ndcg@10', 0.1214), ('hit@10', 0.2147), ('precision@10', 0.0215)])}
  ```
