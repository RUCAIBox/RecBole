# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [NARM](https://recbole.io/docs/user_guide/model/sequential/narm.html)

- **Time cost**: 16521.33s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.005,0.001,0.0005,0.0001]
  num_layers choice [1,2]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  num_layers: 2
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.001, num_layers:2
  Valid result:
  recall@10 : 0.2443    mrr@10 : 0.1321    ndcg@10 : 0.1585    hit@10 : 0.2443    precision@10 : 0.0244
  Test result:
  recall@10 : 0.1929    mrr@10 : 0.1015    ndcg@10 : 0.1229    hit@10 : 0.1929    precision@10 : 0.0193

  learning_rate:0.005, num_layers:2
  Valid result:
  recall@10 : 0.2337    mrr@10 : 0.1122    ndcg@10 : 0.1408    hit@10 : 0.2337    precision@10 : 0.0234
  Test result:
  recall@10 : 0.1873    mrr@10 : 0.0879    ndcg@10 : 0.1112    hit@10 : 0.1873    precision@10 : 0.0187

  learning_rate:0.0005, num_layers:2
  Valid result:
  recall@10 : 0.2415    mrr@10 : 0.13    ndcg@10 : 0.1562    hit@10 : 0.2415    precision@10 : 0.0241
  Test result:
  recall@10 : 0.185    mrr@10 : 0.0977    ndcg@10 : 0.1182    hit@10 : 0.185    precision@10 : 0.0185

  learning_rate:0.0001, num_layers:1
  Valid result:
  recall@10 : 0.2309    mrr@10 : 0.1124    ndcg@10 : 0.1402    hit@10 : 0.2309    precision@10 : 0.0231
  Test result:
  recall@10 : 0.1749    mrr@10 : 0.0827    ndcg@10 : 0.1043    hit@10 : 0.1749    precision@10 : 0.0175

  learning_rate:0.0001, num_layers:2
  Valid result:
  recall@10 : 0.2309    mrr@10 : 0.1124    ndcg@10 : 0.1402    hit@10 : 0.2309    precision@10 : 0.0231
  Test result:
  recall@10 : 0.1749    mrr@10 : 0.0827    ndcg@10 : 0.1043    hit@10 : 0.1749    precision@10 : 0.0175

  learning_rate:0.0005, num_layers:1
  Valid result:
  recall@10 : 0.2415    mrr@10 : 0.13    ndcg@10 : 0.1562    hit@10 : 0.2415    precision@10 : 0.0241
  Test result:
  recall@10 : 0.185    mrr@10 : 0.0977    ndcg@10 : 0.1182    hit@10 : 0.185    precision@10 : 0.0185

  learning_rate:0.001, num_layers:1
  Valid result:
  recall@10 : 0.2443    mrr@10 : 0.1321    ndcg@10 : 0.1585    hit@10 : 0.2443    precision@10 : 0.0244
  Test result:
  recall@10 : 0.1929    mrr@10 : 0.1015    ndcg@10 : 0.1229    hit@10 : 0.1929    precision@10 : 0.0193

  learning_rate:0.005, num_layers:1
  Valid result:
  recall@10 : 0.2337    mrr@10 : 0.1122    ndcg@10 : 0.1408    hit@10 : 0.2337    precision@10 : 0.0234
  Test result:
  recall@10 : 0.1873    mrr@10 : 0.0879    ndcg@10 : 0.1112    hit@10 : 0.1873    precision@10 : 0.0187
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 8/8 [36:42:50<00:00, 16521.33s/trial, best loss: -0.1585]
  best params:  {'learning_rate': 0.001, 'num_layers': 2}
  best result: 
  {'model': 'NARM', 'best_valid_score': 0.1585, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2443), ('mrr@10', 0.1321), ('ndcg@10', 0.1585), ('hit@10', 0.2443), ('precision@10', 0.0244)]), 'test_result': OrderedDict([('recall@10', 0.1929), ('mrr@10', 0.1015), ('ndcg@10', 0.1229), ('hit@10', 0.1929), ('precision@10', 0.0193)])}
  ```
