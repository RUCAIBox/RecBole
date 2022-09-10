# General Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_general.md)

- **Model**: [LightGCN](https://recbole.io/docs/user_guide/model/general/lightgcn.html)

- **Time cost**: 28266.40s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate in [5e-4,1e-3,2e-3] 
  n_layers in [1,2,3,4] 
  reg_weight in [1e-05,1e-04,1e-03,1e-02]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001 
  n_layers: 4  
  reg_weight: 0.01
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0005, n_layers:2, reg_weight:0.01
  Valid result:
  recall@10 : 0.1765    mrr@10 : 0.1263    ndcg@10 : 0.1159    hit@10 : 0.2808    precision@10 : 0.0349
  Test result:
  recall@10 : 0.1809    mrr@10 : 0.1375    ndcg@10 : 0.1241    hit@10 : 0.2853    precision@10 : 0.0366

  learning_rate:0.001, n_layers:4, reg_weight:0.01
  Valid result:
  recall@10 : 0.184     mrr@10 : 0.1344    ndcg@10 : 0.1227    hit@10 : 0.2924    precision@10 : 0.0366
  Test result:
  recall@10 : 0.1902    mrr@10 : 0.1445    ndcg@10 : 0.1306    hit@10 : 0.2979    precision@10 : 0.0386
  
  learning_rate:0.002, n_layers:1, reg_weight:0.01
  Valid result:
  recall@10 : 0.1591    mrr@10 : 0.1141    ndcg@10 : 0.1044    hit@10 : 0.2565    precision@10 : 0.0313
  Test result:
  recall@10 : 0.1638    mrr@10 : 0.1228    ndcg@10 : 0.1108    hit@10 : 0.2626    precision@10 : 0.033
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 48/48 [376:53:07<00:00, 28266.40s/trial, best loss: -0.1227]
  best params:  {'learning_rate': 0.001, 'n_layers': 4, 'reg_weight': 0.01}
  best result: 
  {'model': 'LightGCN', 'best_valid_score': 0.1227, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.184), ('mrr@10', 0.1344), ('ndcg@10', 0.1227), ('hit@10', 0.2924), ('precision@10', 0.0366)]), 'test_result': OrderedDict([('recall@10', 0.1902), ('mrr@10', 0.1445), ('ndcg@10', 0.1306), ('hit@10', 0.2979), ('precision@10', 0.0386)])}
  ```
