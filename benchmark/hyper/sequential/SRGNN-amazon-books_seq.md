# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [SRGNN](https://recbole.io/docs/user_guide/model/sequential/srgnn.html)

- **Time cost**: 13613.66s/trial

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
  learning_rate:0.0005
  Valid result:
  recall@10 : 0.2014    mrr@10 : 0.1116    ndcg@10 : 0.1327    hit@10 : 0.2014    precision@10 : 0.0201
  Test result:
  recall@10 : 0.1535    mrr@10 : 0.0818    ndcg@10 : 0.0986    hit@10 : 0.1535    precision@10 : 0.0153

  learning_rate:0.005
  Valid result:
  recall@10 : 0.2061    mrr@10 : 0.113    ndcg@10 : 0.1349    hit@10 : 0.2061    precision@10 : 0.0206
  Test result:
  recall@10 : 0.1616    mrr@10 : 0.0856    ndcg@10 : 0.1034    hit@10 : 0.1616    precision@10 : 0.0162

  learning_rate:0.003
  Valid result:
  recall@10 : 0.2093    mrr@10 : 0.1123    ndcg@10 : 0.1351    hit@10 : 0.2093    precision@10 : 0.0209
  Test result:
  recall@10 : 0.1594    mrr@10 : 0.084    ndcg@10 : 0.1016    hit@10 : 0.1594    precision@10 : 0.0159

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.1997    mrr@10 : 0.1108    ndcg@10 : 0.1317    hit@10 : 0.1997    precision@10 : 0.02
  Test result:
  recall@10 : 0.1525    mrr@10 : 0.0817    ndcg@10 : 0.0983    hit@10 : 0.1525    precision@10 : 0.0152

  learning_rate:0.001
  Valid result:
  recall@10 : 0.2022    mrr@10 : 0.111    ndcg@10 : 0.1324    hit@10 : 0.2022    precision@10 : 0.0202
  Test result:
  recall@10 : 0.1536    mrr@10 : 0.0812    ndcg@10 : 0.0981    hit@10 : 0.1536    precision@10 : 0.0154
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [18:54:28<00:00, 13613.66s/trial, best loss: -0.1351]
  best params:  {'learning_rate': 0.003}
  best result: 
  {'model': 'SRGNN', 'best_valid_score': 0.1351, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2093), ('mrr@10', 0.1123), ('ndcg@10', 0.1351), ('hit@10', 0.2093), ('precision@10', 0.0209)]), 'test_result': OrderedDict([('recall@10', 0.1594), ('mrr@10', 0.084), ('ndcg@10', 0.1016), ('hit@10', 0.1594), ('precision@10', 0.0159)])}
  ```
