# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [Caser](https://recbole.io/docs/user_guide/model/sequential/caser.html)

- **Time cost**: 34452.71s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0003, 0.0005, 0.001, 0.003, 0.005]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.003
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0003
  Valid result:
  recall@10 : 0.213    mrr@10 : 0.0775    ndcg@10 : 0.1088    hit@10 : 0.213    precision@10 : 0.0213
  Test result:
  recall@10 : 0.1977    mrr@10 : 0.0727    ndcg@10 : 0.1016    hit@10 : 0.1977    precision@10 : 0.0198

  learning_rate:0.003
  Valid result:
  recall@10 : 0.218    mrr@10 : 0.0818    ndcg@10 : 0.1133    hit@10 : 0.218    precision@10 : 0.0218
  Test result:
  recall@10 : 0.2039    mrr@10 : 0.0765    ndcg@10 : 0.1061    hit@10 : 0.2039    precision@10 : 0.0204

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.2142    mrr@10 : 0.0796    ndcg@10 : 0.1108    hit@10 : 0.2142    precision@10 : 0.0214
  Test result:
  recall@10 : 0.2009    mrr@10 : 0.0753    ndcg@10 : 0.1045    hit@10 : 0.2009    precision@10 : 0.0201

  learning_rate:0.005
  Valid result:
  recall@10 : 0.2203    mrr@10 : 0.0806    ndcg@10 : 0.113    hit@10 : 0.2203    precision@10 : 0.022
  Test result:
  recall@10 : 0.1946    mrr@10 : 0.0708    ndcg@10 : 0.0995    hit@10 : 0.1946    precision@10 : 0.0195

  learning_rate:0.001
  Valid result:
  recall@10 : 0.216    mrr@10 : 0.0795    ndcg@10 : 0.111    hit@10 : 0.216    precision@10 : 0.0216
  Test result:
  recall@10 : 0.1989    mrr@10 : 0.076    ndcg@10 : 0.1047    hit@10 : 0.1989    precision@10 : 0.0199
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [47:51:03<00:00, 34452.71s/trial, best loss: -0.1133]
  best params:  {'learning_rate': 0.003}
  best result: 
  {'model': 'Caser', 'best_valid_score': 0.1133, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.218), ('mrr@10', 0.0818), ('ndcg@10', 0.1133), ('hit@10', 0.218), ('precision@10', 0.0218)]), 'test_result': OrderedDict([('recall@10', 0.2039), ('mrr@10', 0.0765), ('ndcg@10', 0.1061), ('hit@10', 0.2039), ('precision@10', 0.0204)])}
  ```
