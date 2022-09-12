# Knowledge-aware Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_kg.md)

- **Model**: [KGIN](https://recbole.io/docs/user_guide/model/knowledge/kgin.html)

- **Time cost**: 4221.58s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [1e-4,1e-3,5e-3]
  node_dropout_rate choice [0.1,0.3,0.5]
  mess_dropout_rate choice [0.0,0.1]
  context_hops choice [2,3]
  n_factors choice [4,8]
  ind choice ['cosine','distance']
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 1e-3
  node_dropout_rate: 0.3
  mess_dropout_rate: 0.0
  context_hops: 2
  n_factors: 4
  ind: 'distance'
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  context_hops:3, ind:cosine, learning_rate:0.001, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.2457    mrr@10 : 0.1327    ndcg@10 : 0.1579    hit@10 : 0.253    precision@10 : 0.0256
  Test result:
  recall@10 : 0.2212    mrr@10 : 0.123    ndcg@10 : 0.1456    hit@10 : 0.2245    precision@10 : 0.0227
  
  context_hops:2, ind:distance, learning_rate:0.001, mess_dropout_rate:0.0, n_factors:8, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.2752    mrr@10 : 0.1512    ndcg@10 : 0.179    hit@10 : 0.2827    precision@10 : 0.0287
  Test result:
  recall@10 : 0.2437    mrr@10 : 0.1423    ndcg@10 : 0.1656    hit@10 : 0.247    precision@10 : 0.0249
  
  context_hops:2, ind:cosine, learning_rate:0.0001, mess_dropout_rate:0.0, n_factors:4, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.0749    mrr@10 : 0.0425    ndcg@10 : 0.0496    hit@10 : 0.0776    precision@10 : 0.0078
  Test result:
  recall@10 : 0.0664    mrr@10 : 0.0349    ndcg@10 : 0.042    hit@10 : 0.0677    precision@10 : 0.0068
  
  context_hops:3, ind:distance, learning_rate:0.0001, mess_dropout_rate:0.1, n_factors:4, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.1162    mrr@10 : 0.0557    ndcg@10 : 0.069    hit@10 : 0.1203    precision@10 : 0.0121
  Test result:
  recall@10 : 0.0914    mrr@10 : 0.0459    ndcg@10 : 0.0561    hit@10 : 0.0932    precision@10 : 0.0094
  
  context_hops:3, ind:distance, learning_rate:0.001, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.2544    mrr@10 : 0.1401    ndcg@10 : 0.1657    hit@10 : 0.2615    precision@10 : 0.0265
  Test result:
  recall@10 : 0.224    mrr@10 : 0.1283    ndcg@10 : 0.1502    hit@10 : 0.2275    precision@10 : 0.023
  
  context_hops:2, ind:distance, learning_rate:0.0001, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.1487    mrr@10 : 0.0796    ndcg@10 : 0.0948    hit@10 : 0.1532    precision@10 : 0.0154
  Test result:
  recall@10 : 0.1228    mrr@10 : 0.0676    ndcg@10 : 0.0801    hit@10 : 0.1249    precision@10 : 0.0125
  
  context_hops:2, ind:cosine, learning_rate:0.005, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.269    mrr@10 : 0.1483    ndcg@10 : 0.1756    hit@10 : 0.2758    precision@10 : 0.028
  Test result:
  recall@10 : 0.2442    mrr@10 : 0.1425    ndcg@10 : 0.166    hit@10 : 0.2474    precision@10 : 0.0249
  
  context_hops:2, ind:distance, learning_rate:0.005, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.2479    mrr@10 : 0.1362    ndcg@10 : 0.1614    hit@10 : 0.2541    precision@10 : 0.0258
  Test result:
  recall@10 : 0.2259    mrr@10 : 0.1323    ndcg@10 : 0.1538    hit@10 : 0.229    precision@10 : 0.0231
  
  context_hops:3, ind:cosine, learning_rate:0.001, mess_dropout_rate:0.1, n_factors:4, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.2548    mrr@10 : 0.1364    ndcg@10 : 0.1628    hit@10 : 0.2621    precision@10 : 0.0266
  Test result:
  recall@10 : 0.2283    mrr@10 : 0.1303    ndcg@10 : 0.1527    hit@10 : 0.2319    precision@10 : 0.0234
  
  context_hops:3, ind:distance, learning_rate:0.0001, mess_dropout_rate:0.1, n_factors:4, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.1477    mrr@10 : 0.0817    ndcg@10 : 0.0963    hit@10 : 0.1526    precision@10 : 0.0154
  Test result:
  recall@10 : 0.1197    mrr@10 : 0.0659    ndcg@10 : 0.0779    hit@10 : 0.1219    precision@10 : 0.0122
  
  context_hops:2, ind:distance, learning_rate:0.001, mess_dropout_rate:0.0, n_factors:8, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.2712    mrr@10 : 0.1457    ndcg@10 : 0.1739    hit@10 : 0.2787    precision@10 : 0.0283
  Test result:
  recall@10 : 0.2406    mrr@10 : 0.1378    ndcg@10 : 0.1615    hit@10 : 0.244    precision@10 : 0.0246
  
  context_hops:2, ind:cosine, learning_rate:0.001, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.2728    mrr@10 : 0.1478    ndcg@10 : 0.176    hit@10 : 0.2798    precision@10 : 0.0284
  Test result:
  recall@10 : 0.2463    mrr@10 : 0.1419    ndcg@10 : 0.166    hit@10 : 0.25    precision@10 : 0.0252
  
  context_hops:2, ind:cosine, learning_rate:0.001, mess_dropout_rate:0.1, n_factors:4, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.2699    mrr@10 : 0.1456    ndcg@10 : 0.1735    hit@10 : 0.277    precision@10 : 0.0281
  Test result:
  recall@10 : 0.244    mrr@10 : 0.141    ndcg@10 : 0.1648    hit@10 : 0.2474    precision@10 : 0.0249
  
  context_hops:3, ind:distance, learning_rate:0.001, mess_dropout_rate:0.0, n_factors:8, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.2324    mrr@10 : 0.1265    ndcg@10 : 0.1503    hit@10 : 0.2395    precision@10 : 0.0243
  Test result:
  recall@10 : 0.2024    mrr@10 : 0.1155    ndcg@10 : 0.1354    hit@10 : 0.2056    precision@10 : 0.0208
  ```


- **Logging Result**:

  ```yaml
  100%|██████████| 144/144 [168:51:48<00:00, 4221.58s/trial, best loss: -0.1792]
  best params:  {'context_hops': 2, 'ind': 'distance', 'learning_rate': 0.001, 'mess_dropout_rate': 0.0, 'n_factors': 4, 'node_dropout_rate': 0.3}
  best result: 
  {'model': 'KGIN', 'best_valid_score': 0.1792, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2747), ('mrr@10', 0.1515), ('ndcg@10', 0.1792), ('hit@10', 0.2823), ('precision@10', 0.0287)]), 'test_result': OrderedDict([('recall@10', 0.2442), ('mrr@10', 0.1438), ('ndcg@10', 0.1669), ('hit@10', 0.2476), ('precision@10', 0.025)])}
  ```
  
  
