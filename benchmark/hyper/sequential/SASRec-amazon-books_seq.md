# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [SASRec](https://recbole.io/docs/user_guide/model/sequential/sasrec.html)

- **Time cost**: 15169.40s/trial

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
  learning_rate:0.003
  Valid result:
  recall@10 : 0.2397    mrr@10 : 0.0991    ndcg@10 : 0.1323    hit@10 : 0.2397    precision@10 : 0.024
  Test result:
  recall@10 : 0.1943    mrr@10 : 0.0805    ndcg@10 : 0.1073    hit@10 : 0.1943    precision@10 : 0.0194

  learning_rate:0.001
  Valid result:
  recall@10 : 0.2441    mrr@10 : 0.102    ndcg@10 : 0.1356    hit@10 : 0.2441    precision@10 : 0.0244
  Test result:
  recall@10 : 0.1993    mrr@10 : 0.0838    ndcg@10 : 0.111    hit@10 : 0.1993    precision@10 : 0.0199

  learning_rate:0.005
  Valid result:
  recall@10 : 0.2299    mrr@10 : 0.0965    ndcg@10 : 0.1279    hit@10 : 0.2299    precision@10 : 0.023
  Test result:
  recall@10 : 0.1861    mrr@10 : 0.0773    ndcg@10 : 0.1028    hit@10 : 0.1861    precision@10 : 0.0186

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.2501    mrr@10 : 0.1031    ndcg@10 : 0.1378    hit@10 : 0.2501    precision@10 : 0.025
  Test result:
  recall@10 : 0.2049    mrr@10 : 0.0849    ndcg@10 : 0.1131    hit@10 : 0.2049    precision@10 : 0.0205

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.2475    mrr@10 : 0.1017    ndcg@10 : 0.1362    hit@10 : 0.2475    precision@10 : 0.0248
  Test result:
  recall@10 : 0.2035    mrr@10 : 0.0839    ndcg@10 : 0.112    hit@10 : 0.2035    precision@10 : 0.0204
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [21:04:07<00:00, 15169.40s/trial, best loss: -0.1378]
  best params:  {'learning_rate': 0.0003}
  best result: 
  {'model': 'SASRec', 'best_valid_score': 0.1378, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2501), ('mrr@10', 0.1031), ('ndcg@10', 0.1378), ('hit@10', 0.2501), ('precision@10', 0.025)]), 'test_result': OrderedDict([('recall@10', 0.2049), ('mrr@10', 0.0849), ('ndcg@10', 0.1131), ('hit@10', 0.2049), ('precision@10', 0.0205)])}
  ```
