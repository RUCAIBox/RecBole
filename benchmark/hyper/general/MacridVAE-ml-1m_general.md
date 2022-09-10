# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [MacridVAE](https://recbole.io/docs/user_guide/model/general/macridvae.html)

- **Time cost**: 152.70s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate in [0.0005,0.001,0.005,0.01,0.05]
  kafc in [3,5,10,20]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.005  
  kafc: 20
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  kafc:20, learning_rate:0.0005
  Valid result:
  recall@10 : 0.1602    mrr@10 : 0.3475    ndcg@10 : 0.1943    hit@10 : 0.697     precision@10 : 0.1438
  Test result:
  recall@10 : 0.1747    mrr@10 : 0.416     ndcg@10 : 0.2355    hit@10 : 0.7253    precision@10 : 0.1743

  kafc:20, learning_rate:0.005
  Valid result:
  recall@10 : 0.1587    mrr@10 : 0.353     ndcg@10 : 0.1945    hit@10 : 0.7023    precision@10 : 0.142
  Test result:
  recall@10 : 0.1719    mrr@10 : 0.402     ndcg@10 : 0.2272    hit@10 : 0.7154    precision@10 : 0.1691
  
  kafc:3, learning_rate:0.0005
  Valid result:
  recall@10 : 0.1592    mrr@10 : 0.3471    ndcg@10 : 0.1934    hit@10 : 0.7005    precision@10 : 0.1437
  Test result:
  recall@10 : 0.1783    mrr@10 : 0.4129    ndcg@10 : 0.2356    hit@10 : 0.7321    precision@10 : 0.1746
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 20/20 [50:53<00:00, 152.70s/trial, best loss: -0.1945]
  best params:  {'kafc': 20, 'learning_rate': 0.005}
  best result: 
  {'model': 'MacridVAE', 'best_valid_score': 0.1945, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1587), ('mrr@10', 0.353), ('ndcg@10', 0.1945), ('hit@10', 0.7023), ('precision@10', 0.142)]), 'test_result': OrderedDict([('recall@10', 0.1719), ('mrr@10', 0.402), ('ndcg@10', 0.2272), ('hit@10', 0.7154), ('precision@10', 0.1691)])}
  ```
