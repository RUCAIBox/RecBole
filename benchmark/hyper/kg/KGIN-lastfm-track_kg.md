# Knowledge-aware Recommendation

- **Dataset**: [Lastfm-track](../../md/lastfm-track_kg.md)

- **Model**: [KGIN](https://recbole.io/docs/user_guide/model/knowledge/kgin.html)

- **Time cost**: 52128.97s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [1e-4,1e-3]
  node_dropout_rate choice [0.1,0.3,0.5]
  mess_dropout_rate choice [0.1]
  context_hops choice [3]
  n_factors choice [8]
  ind choice ['cosine','distance']
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 1e-3
  node_dropout_rate: 0.5
  mess_dropout_rate: 0.1
  context_hops: 3
  n_factors: 8
  ind: 'cosine'
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  context_hops:3, ind:cosine, learning_rate:0.0001, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.08    mrr@10 : 0.0831    ndcg@10 : 0.0596    hit@10 : 0.1968    precision@10 : 0.0248
  Test result:
  recall@10 : 0.078    mrr@10 : 0.0865    ndcg@10 : 0.0586    hit@10 : 0.2082    precision@10 : 0.0262
  
  context_hops:3, ind:cosine, learning_rate:0.001, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.1871    mrr@10 : 0.1997    ndcg@10 : 0.1461    hit@10 : 0.4301    precision@10 : 0.063
  Test result:
  recall@10 : 0.1921    mrr@10 : 0.2188    ndcg@10 : 0.1586    hit@10 : 0.4299    precision@10 : 0.0662
  
  context_hops:3, ind:distance, learning_rate:0.001, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.1722    mrr@10 : 0.1863    ndcg@10 : 0.1339    hit@10 : 0.4081    precision@10 : 0.0589
  Test result:
  recall@10 : 0.1775    mrr@10 : 0.2026    ndcg@10 : 0.1455    hit@10 : 0.4076    precision@10 : 0.0621
  
  context_hops:3, ind:cosine, learning_rate:0.0001, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.0901    mrr@10 : 0.0918    ndcg@10 : 0.0668    hit@10 : 0.2141    precision@10 : 0.0271
  Test result:
  recall@10 : 0.0856    mrr@10 : 0.0922    ndcg@10 : 0.0637    hit@10 : 0.2203    precision@10 : 0.0278
  
  context_hops:3, ind:distance, learning_rate:0.0001, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.08    mrr@10 : 0.0831    ndcg@10 : 0.0596    hit@10 : 0.1968    precision@10 : 0.0248
  Test result:
  recall@10 : 0.0777    mrr@10 : 0.0862    ndcg@10 : 0.0584    hit@10 : 0.2076    precision@10 : 0.0261
  
  context_hops:3, ind:distance, learning_rate:0.001, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.1865    mrr@10 : 0.1997    ndcg@10 : 0.1458    hit@10 : 0.4302    precision@10 : 0.0628
  Test result:
  recall@10 : 0.1936    mrr@10 : 0.2179    ndcg@10 : 0.1592    hit@10 : 0.4311    precision@10 : 0.0666
  
  context_hops:3, ind:distance, learning_rate:0.001, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.1809    mrr@10 : 0.195    ndcg@10 : 0.1416    hit@10 : 0.4216    precision@10 : 0.0614
  Test result:
  recall@10 : 0.1875    mrr@10 : 0.2136    ndcg@10 : 0.1539    hit@10 : 0.4228    precision@10 : 0.065
  
  context_hops:3, ind:cosine, learning_rate:0.0001, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.1377    mrr@10 : 0.1637    ndcg@10 : 0.1115    hit@10 : 0.3525    precision@10 : 0.0497
  Test result:
  recall@10 : 0.1426    mrr@10 : 0.1766    ndcg@10 : 0.1195    hit@10 : 0.3582    precision@10 : 0.0529
  
  context_hops:3, ind:cosine, learning_rate:0.001, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.175    mrr@10 : 0.1864    ndcg@10 : 0.1354    hit@10 : 0.4098    precision@10 : 0.0591
  Test result:
  recall@10 : 0.1796    mrr@10 : 0.2039    ndcg@10 : 0.1468    hit@10 : 0.4094    precision@10 : 0.0624
  
  context_hops:3, ind:cosine, learning_rate:0.001, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.1812    mrr@10 : 0.1959    ndcg@10 : 0.142    hit@10 : 0.4225    precision@10 : 0.0615
  Test result:
  recall@10 : 0.1869    mrr@10 : 0.2129    ndcg@10 : 0.1535    hit@10 : 0.4215    precision@10 : 0.0648
  
  context_hops:3, ind:distance, learning_rate:0.0001, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.1381    mrr@10 : 0.164    ndcg@10 : 0.1117    hit@10 : 0.3542    precision@10 : 0.0501
  Test result:
  recall@10 : 0.1421    mrr@10 : 0.1768    ndcg@10 : 0.1194    hit@10 : 0.3579    precision@10 : 0.0528
  
  context_hops:3, ind:distance, learning_rate:0.0001, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.09    mrr@10 : 0.0917    ndcg@10 : 0.0667    hit@10 : 0.2141    precision@10 : 0.0271
  Test result:
  recall@10 : 0.0853    mrr@10 : 0.0921    ndcg@10 : 0.0636    hit@10 : 0.2198    precision@10 : 0.0277
  ```


- **Logging Result**:

  ```yaml
  100%|██████████| 12/12 [173:45:47<00:00, 52128.97s/trial, best loss: -0.1461]
  best params:  {'context_hops': 3, 'ind': 'cosine', 'learning_rate': 0.001, 'mess_dropout_rate': 0.1, 'n_factors': 8, 'node_dropout_rate': 0.5}
  best result: 
  {'model': 'KGIN', 'best_valid_score': 0.1461, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1871), ('mrr@10', 0.1997), ('ndcg@10', 0.1461), ('hit@10', 0.4301), ('precision@10', 0.063)]), 'test_result': OrderedDict([('recall@10', 0.1921), ('mrr@10', 0.2188), ('ndcg@10', 0.1586), ('hit@10', 0.4299), ('precision@10', 0.0662)])}
  ```
  
  
