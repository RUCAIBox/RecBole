# Knowledge-aware Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_kg.md)

- **Model**: [MCCLK](https://recbole.io/docs/user_guide/model/knowledge/mcclk.html)

- **Time cost**: 13132.76s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [1e-4,1e-3,5e-3]
  node_dropout_rate choice [0.1,0.3,0.5]
  mess_dropout_rate choice [0.0,0.1]
  build_graph_separately choice [True, False]
  loss_type choice ['BPR']
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 1e-3
  node_dropout_rate: 0.5
  mess_dropout_rate: 0.0
  build_graph_separately: True
  loss_type: 'BPR'
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  build_graph_separately:True, learning_rate:0.0001, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.2611    mrr@10 : 0.135    ndcg@10 : 0.1635    hit@10 : 0.2677    precision@10 : 0.027
  Test result:
  recall@10 : 0.2401    mrr@10 : 0.1287    ndcg@10 : 0.1543    hit@10 : 0.2433    precision@10 : 0.0245
  
  build_graph_separately:True, learning_rate:0.0001, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.2635    mrr@10 : 0.1367    ndcg@10 : 0.1654    hit@10 : 0.27    precision@10 : 0.0273
  Test result:
  recall@10 : 0.243    mrr@10 : 0.1319    ndcg@10 : 0.1574    hit@10 : 0.2463    precision@10 : 0.0248
  
  build_graph_separately:True, learning_rate:0.001, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.2668    mrr@10 : 0.1501    ndcg@10 : 0.1762    hit@10 : 0.2739    precision@10 : 0.0279
  Test result:
  recall@10 : 0.2342    mrr@10 : 0.1431    ndcg@10 : 0.1639    hit@10 : 0.2379    precision@10 : 0.024
  
  build_graph_separately:True, learning_rate:0.005, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.2607    mrr@10 : 0.147    ndcg@10 : 0.1727    hit@10 : 0.2673    precision@10 : 0.0271
  Test result:
  recall@10 : 0.2288    mrr@10 : 0.1403    ndcg@10 : 0.1606    hit@10 : 0.2321    precision@10 : 0.0234
  
  build_graph_separately:True, learning_rate:0.005, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.2659    mrr@10 : 0.15    ndcg@10 : 0.1761    hit@10 : 0.2728    precision@10 : 0.0277
  Test result:
  recall@10 : 0.2329    mrr@10 : 0.1416    ndcg@10 : 0.1626    hit@10 : 0.2363    precision@10 : 0.0238
  
  build_graph_separately:True, learning_rate:0.005, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.2446    mrr@10 : 0.1402    ndcg@10 : 0.1635    hit@10 : 0.2514    precision@10 : 0.0255
  Test result:
  recall@10 : 0.216    mrr@10 : 0.1323    ndcg@10 : 0.1515    hit@10 : 0.219    precision@10 : 0.0221
  
  build_graph_separately:False, learning_rate:0.001, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.2781    mrr@10 : 0.1558    ndcg@10 : 0.1831    hit@10 : 0.2858    precision@10 : 0.0291
  Test result:
  recall@10 : 0.2492    mrr@10 : 0.1505    ndcg@10 : 0.1732    hit@10 : 0.2528    precision@10 : 0.0255
  
  build_graph_separately:False, learning_rate:0.001, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.2664    mrr@10 : 0.1495    ndcg@10 : 0.1758    hit@10 : 0.274    precision@10 : 0.0278
  Test result:
  recall@10 : 0.2375    mrr@10 : 0.1433    ndcg@10 : 0.1649    hit@10 : 0.2414    precision@10 : 0.0244
  
  build_graph_separately:False, learning_rate:0.005, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.2603    mrr@10 : 0.1469    ndcg@10 : 0.1723    hit@10 : 0.2672    precision@10 : 0.0271
  Test result:
  recall@10 : 0.2288    mrr@10 : 0.1395    ndcg@10 : 0.1599    hit@10 : 0.232    precision@10 : 0.0234
  
  build_graph_separately:True, learning_rate:0.001, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.2841    mrr@10 : 0.1585    ndcg@10 : 0.1867    hit@10 : 0.2914    precision@10 : 0.0296
  Test result:
  recall@10 : 0.2546    mrr@10 : 0.1521    ndcg@10 : 0.1757    hit@10 : 0.2583    precision@10 : 0.0261
  
  build_graph_separately:False, learning_rate:0.005, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.2644    mrr@10 : 0.15    ndcg@10 : 0.1755    hit@10 : 0.2715    precision@10 : 0.0275
  Test result:
  recall@10 : 0.2368    mrr@10 : 0.144    ndcg@10 : 0.1653    hit@10 : 0.2401    precision@10 : 0.0242
  
  build_graph_separately:True, learning_rate:0.001, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.2792    mrr@10 : 0.1524    ndcg@10 : 0.181    hit@10 : 0.2863    precision@10 : 0.0291
  Test result:
  recall@10 : 0.2505    mrr@10 : 0.1468    ndcg@10 : 0.1706    hit@10 : 0.2542    precision@10 : 0.0256
  
  build_graph_separately:True, learning_rate:0.0001, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.2569    mrr@10 : 0.1353    ndcg@10 : 0.1627    hit@10 : 0.2636    precision@10 : 0.0266
  Test result:
  recall@10 : 0.2323    mrr@10 : 0.1288    ndcg@10 : 0.1526    hit@10 : 0.2356    precision@10 : 0.0237
  
  build_graph_separately:False, learning_rate:0.001, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.2785    mrr@10 : 0.1556    ndcg@10 : 0.1831    hit@10 : 0.2861    precision@10 : 0.0291
  Test result:
  recall@10 : 0.2501    mrr@10 : 0.1497    ndcg@10 : 0.1728    hit@10 : 0.2538    precision@10 : 0.0256
  
  build_graph_separately:False, learning_rate:0.001, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.2825    mrr@10 : 0.1552    ndcg@10 : 0.1837    hit@10 : 0.2901    precision@10 : 0.0294
  Test result:
  recall@10 : 0.2526    mrr@10 : 0.1504    ndcg@10 : 0.1739    hit@10 : 0.2562    precision@10 : 0.0259
  
  build_graph_separately:False, learning_rate:0.005, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.2681    mrr@10 : 0.1494    ndcg@10 : 0.1762    hit@10 : 0.275    precision@10 : 0.0279
  Test result:
  recall@10 : 0.2376    mrr@10 : 0.1434    ndcg@10 : 0.165    hit@10 : 0.2409    precision@10 : 0.0243
  
  build_graph_separately:False, learning_rate:0.005, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.2453    mrr@10 : 0.1404    ndcg@10 : 0.164    hit@10 : 0.2518    precision@10 : 0.0255
  Test result:
  recall@10 : 0.2176    mrr@10 : 0.134    ndcg@10 : 0.1533    hit@10 : 0.2207    precision@10 : 0.0223
  
  build_graph_separately:False, learning_rate:0.0001, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.2598    mrr@10 : 0.1344    ndcg@10 : 0.1628    hit@10 : 0.2662    precision@10 : 0.0269
  Test result:
  recall@10 : 0.2395    mrr@10 : 0.1285    ndcg@10 : 0.154    hit@10 : 0.2428    precision@10 : 0.0244
  
  build_graph_separately:True, learning_rate:0.001, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.2657    mrr@10 : 0.1506    ndcg@10 : 0.1765    hit@10 : 0.2726    precision@10 : 0.0278
  Test result:
  recall@10 : 0.2351    mrr@10 : 0.1422    ndcg@10 : 0.1635    hit@10 : 0.2387    precision@10 : 0.0241
  
  build_graph_separately:True, learning_rate:0.005, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.2669    mrr@10 : 0.1506    ndcg@10 : 0.1769    hit@10 : 0.2738    precision@10 : 0.0278
  Test result:
  recall@10 : 0.2353    mrr@10 : 0.1431    ndcg@10 : 0.1643    hit@10 : 0.2387    precision@10 : 0.0241
  
  build_graph_separately:True, learning_rate:0.0001, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.2591    mrr@10 : 0.1341    ndcg@10 : 0.1624    hit@10 : 0.2656    precision@10 : 0.0268
  Test result:
  recall@10 : 0.2403    mrr@10 : 0.1296    ndcg@10 : 0.1551    hit@10 : 0.2436    precision@10 : 0.0245
  ```


- **Logging Result**:

  ```yaml
  100%|██████████| 36/36 [131:19:39<00:00, 13132.76s/trial, best loss: -0.1867]
  best params:  {'build_graph_separately': True, 'learning_rate': 0.001, 'loss_type': 'BPR', 'mess_dropout_rate': 0.0, 'node_dropout_rate': 0.5}
  best result: 
  {'model': 'MCCLK', 'best_valid_score': 0.1867, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2841), ('mrr@10', 0.1585), ('ndcg@10', 0.1867), ('hit@10', 0.2914), ('precision@10', 0.0296)]), 'test_result': OrderedDict([('recall@10', 0.2546), ('mrr@10', 0.1521), ('ndcg@10', 0.1757), ('hit@10', 0.2583), ('precision@10', 0.0261)])}
  ```
  
  
