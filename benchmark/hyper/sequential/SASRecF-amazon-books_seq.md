# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [SASRecF](https://recbole.io/docs/user_guide/model/sequential/sasrecf.html)

- **Time cost**: 18129.82s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0003, 0.0005, 0.001, 0.003, 0.005]
  ```

- **Best parameters**:

  ```
  learning_rate: 0.0003
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.005
  Valid result:
  recall@10 : 0.221    mrr@10 : 0.1182    ndcg@10 : 0.1423    hit@10 : 0.221    precision@10 : 0.0221
  Test result:
  recall@10 : 0.1781    mrr@10 : 0.092    ndcg@10 : 0.1121    hit@10 : 0.1781    precision@10 : 0.0178

  learning_rate:0.001
  Valid result:
  recall@10 : 0.2311    mrr@10 : 0.1234    ndcg@10 : 0.1486    hit@10 : 0.2311    precision@10 : 0.0231
  Test result:
  recall@10 : 0.1864    mrr@10 : 0.0965    ndcg@10 : 0.1175    hit@10 : 0.1864    precision@10 : 0.0186

  learning_rate:0.003
  Valid result:
  recall@10 : 0.2259    mrr@10 : 0.1217    ndcg@10 : 0.1461    hit@10 : 0.2259    precision@10 : 0.0226
  Test result:
  recall@10 : 0.1796    mrr@10 : 0.0932    ndcg@10 : 0.1134    hit@10 : 0.1796    precision@10 : 0.018

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.2299    mrr@10 : 0.1245    ndcg@10 : 0.1492    hit@10 : 0.2299    precision@10 : 0.023
  Test result:
  recall@10 : 0.1834    mrr@10 : 0.096    ndcg@10 : 0.1165    hit@10 : 0.1834    precision@10 : 0.0183

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.2316    mrr@10 : 0.1252    ndcg@10 : 0.1501    hit@10 : 0.2316    precision@10 : 0.0232
  Test result:
  recall@10 : 0.1866    mrr@10 : 0.098    ndcg@10 : 0.1188    hit@10 : 0.1866    precision@10 : 0.0187
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [25:10:49<00:00, 18129.82s/trial, best loss: -0.1501]
  best params:  {'learning_rate': 0.0003}
  best result: 
  {'model': 'SASRecF', 'best_valid_score': 0.1501, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2316), ('mrr@10', 0.1252), ('ndcg@10', 0.1501), ('hit@10', 0.2316), ('precision@10', 0.0232)]), 'test_result': OrderedDict([('recall@10', 0.1866), ('mrr@10', 0.098), ('ndcg@10', 0.1188), ('hit@10', 0.1866), ('precision@10', 0.0187)])}
  ```
