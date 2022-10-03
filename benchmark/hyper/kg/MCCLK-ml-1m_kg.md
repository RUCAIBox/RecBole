# Knowledge-aware Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_kg.md)

- **Model**: [MCCLK](https://recbole.io/docs/user_guide/model/knowledge/mcclk.html)

- **Time cost**: 43701.79s/trial

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
  mess_dropout_rate: 0.1
  build_graph_separately: True
  loss_type: 'BPR'
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  build_graph_separately:False, learning_rate:0.001, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.161    mrr@10 : 0.3623    ndcg@10 : 0.1997    hit@10 : 0.7078    precision@10 : 0.1483
  Test result:
  recall@10 : 0.1821    mrr@10 : 0.4239    ndcg@10 : 0.245    hit@10 : 0.7311    precision@10 : 0.1825
  
  build_graph_separately:False, learning_rate:0.001, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.1626    mrr@10 : 0.3643    ndcg@10 : 0.2018    hit@10 : 0.7159    precision@10 : 0.1501
  Test result:
  recall@10 : 0.1817    mrr@10 : 0.4248    ndcg@10 : 0.2456    hit@10 : 0.733    precision@10 : 0.1829
  
  build_graph_separately:False, learning_rate:0.0001, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.1589    mrr@10 : 0.3587    ndcg@10 : 0.1986    hit@10 : 0.706    precision@10 : 0.1487
  Test result:
  recall@10 : 0.1766    mrr@10 : 0.4198    ndcg@10 : 0.2425    hit@10 : 0.7187    precision@10 : 0.1802
  
  build_graph_separately:False, learning_rate:0.005, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.1555    mrr@10 : 0.3438    ndcg@10 : 0.1903    hit@10 : 0.7005    precision@10 : 0.1433
  Test result:
  recall@10 : 0.1693    mrr@10 : 0.3868    ndcg@10 : 0.2182    hit@10 : 0.7118    precision@10 : 0.1611
  
  build_graph_separately:True, learning_rate:0.0001, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.1544    mrr@10 : 0.3499    ndcg@10 : 0.1933    hit@10 : 0.6942    precision@10 : 0.1454
  Test result:
  recall@10 : 0.1714    mrr@10 : 0.415    ndcg@10 : 0.2382    hit@10 : 0.7089    precision@10 : 0.1771
  
  build_graph_separately:False, learning_rate:0.005, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.1544    mrr@10 : 0.3477    ndcg@10 : 0.1908    hit@10 : 0.6977    precision@10 : 0.1429
  Test result:
  recall@10 : 0.1672    mrr@10 : 0.3837    ndcg@10 : 0.2165    hit@10 : 0.7113    precision@10 : 0.1608
  
  build_graph_separately:False, learning_rate:0.001, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.1605    mrr@10 : 0.3618    ndcg@10 : 0.1994    hit@10 : 0.7074    precision@10 : 0.1484
  Test result:
  recall@10 : 0.1809    mrr@10 : 0.4194    ndcg@10 : 0.2425    hit@10 : 0.7277    precision@10 : 0.1805
  
  build_graph_separately:True, learning_rate:0.001, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.1612    mrr@10 : 0.3642    ndcg@10 : 0.2004    hit@10 : 0.7098    precision@10 : 0.1485
  Test result:
  recall@10 : 0.1802    mrr@10 : 0.4213    ndcg@10 : 0.2438    hit@10 : 0.7265    precision@10 : 0.1817
  
  build_graph_separately:True, learning_rate:0.005, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.1552    mrr@10 : 0.3471    ndcg@10 : 0.1911    hit@10 : 0.6972    precision@10 : 0.1431
  Test result:
  recall@10 : 0.1719    mrr@10 : 0.3946    ndcg@10 : 0.2234    hit@10 : 0.7137    precision@10 : 0.1658
  
  build_graph_separately:True, learning_rate:0.005, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.1577    mrr@10 : 0.3486    ndcg@10 : 0.1918    hit@10 : 0.6997    precision@10 : 0.1427
  Test result:
  recall@10 : 0.1701    mrr@10 : 0.3861    ndcg@10 : 0.2171    hit@10 : 0.7099    precision@10 : 0.1599
  
  build_graph_separately:True, learning_rate:0.005, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.1571    mrr@10 : 0.3518    ndcg@10 : 0.1925    hit@10 : 0.705    precision@10 : 0.1432
  Test result:
  recall@10 : 0.1687    mrr@10 : 0.3847    ndcg@10 : 0.2165    hit@10 : 0.7058    precision@10 : 0.1599
  
  build_graph_separately:True, learning_rate:0.001, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.159    mrr@10 : 0.3586    ndcg@10 : 0.1982    hit@10 : 0.7053    precision@10 : 0.148
  Test result:
  recall@10 : 0.1768    mrr@10 : 0.4157    ndcg@10 : 0.2399    hit@10 : 0.7195    precision@10 : 0.1786
  
  build_graph_separately:True, learning_rate:0.005, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.1525    mrr@10 : 0.3472    ndcg@10 : 0.1905    hit@10 : 0.69    precision@10 : 0.1431
  Test result:
  recall@10 : 0.1692    mrr@10 : 0.3834    ndcg@10 : 0.2182    hit@10 : 0.7122    precision@10 : 0.1624
  
  build_graph_separately:False, learning_rate:0.0001, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.0641    mrr@10 : 0.1538    ndcg@10 : 0.0813    hit@10 : 0.4341    precision@10 : 0.0715
  Test result:
  recall@10 : 0.0484    mrr@10 : 0.132    ndcg@10 : 0.0704    hit@10 : 0.3671    precision@10 : 0.0653
  
  build_graph_separately:False, learning_rate:0.005, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.1546    mrr@10 : 0.3474    ndcg@10 : 0.19    hit@10 : 0.6967    precision@10 : 0.143
  Test result:
  recall@10 : 0.17    mrr@10 : 0.3918    ndcg@10 : 0.2212    hit@10 : 0.7104    precision@10 : 0.1631
  
  build_graph_separately:False, learning_rate:0.005, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.1549    mrr@10 : 0.3399    ndcg@10 : 0.1889    hit@10 : 0.6987    precision@10 : 0.1434
  Test result:
  recall@10 : 0.1682    mrr@10 : 0.3867    ndcg@10 : 0.2175    hit@10 : 0.7098    precision@10 : 0.1616
  
  build_graph_separately:False, learning_rate:0.0001, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.1581    mrr@10 : 0.3629    ndcg@10 : 0.1993    hit@10 : 0.7046    precision@10 : 0.1487
  Test result:
  recall@10 : 0.1791    mrr@10 : 0.4257    ndcg@10 : 0.2455    hit@10 : 0.725    precision@10 : 0.1817
  
  build_graph_separately:True, learning_rate:0.0001, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.0406    mrr@10 : 0.1503    ndcg@10 : 0.0622    hit@10 : 0.3317    precision@10 : 0.0471
  Test result:
  recall@10 : 0.02    mrr@10 : 0.1036    ndcg@10 : 0.0421    hit@10 : 0.2256    precision@10 : 0.035
  
  build_graph_separately:True, learning_rate:0.001, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.1634    mrr@10 : 0.3644    ndcg@10 : 0.2019    hit@10 : 0.7127    precision@10 : 0.1491
  Test result:
  recall@10 : 0.1832    mrr@10 : 0.4242    ndcg@10 : 0.2465    hit@10 : 0.7341    precision@10 : 0.1829
  
  build_graph_separately:False, learning_rate:0.005, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.1554    mrr@10 : 0.3466    ndcg@10 : 0.1905    hit@10 : 0.7005    precision@10 : 0.1426
  Test result:
  recall@10 : 0.1681    mrr@10 : 0.385    ndcg@10 : 0.2175    hit@10 : 0.7045    precision@10 : 0.1601
  
  build_graph_separately:False, learning_rate:0.0001, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.1466    mrr@10 : 0.3419    ndcg@10 : 0.1862    hit@10 : 0.6811    precision@10 : 0.1408
  Test result:
  recall@10 : 0.1606    mrr@10 : 0.4063    ndcg@10 : 0.2286    hit@10 : 0.6909    precision@10 : 0.171
  
  build_graph_separately:True, learning_rate:0.0001, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.1559    mrr@10 : 0.3575    ndcg@10 : 0.1963    hit@10 : 0.703    precision@10 : 0.1466
  Test result:
  recall@10 : 0.175    mrr@10 : 0.4136    ndcg@10 : 0.2399    hit@10 : 0.7167    precision@10 : 0.1787
  
  build_graph_separately:True, learning_rate:0.001, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.1596    mrr@10 : 0.3602    ndcg@10 : 0.1987    hit@10 : 0.7064    precision@10 : 0.1481
  Test result:
  recall@10 : 0.1809    mrr@10 : 0.4246    ndcg@10 : 0.2438    hit@10 : 0.7257    precision@10 : 0.1801
  
  build_graph_separately:True, learning_rate:0.005, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.153    mrr@10 : 0.347    ndcg@10 : 0.191    hit@10 : 0.6907    precision@10 : 0.1432
  Test result:
  recall@10 : 0.1683    mrr@10 : 0.3841    ndcg@10 : 0.2171    hit@10 : 0.7136    precision@10 : 0.1618
  
  build_graph_separately:True, learning_rate:0.0001, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.1483    mrr@10 : 0.3447    ndcg@10 : 0.1877    hit@10 : 0.6851    precision@10 : 0.1417
  Test result:
  recall@10 : 0.1632    mrr@10 : 0.4047    ndcg@10 : 0.2309    hit@10 : 0.6935    precision@10 : 0.1732
  
  build_graph_separately:False, learning_rate:0.001, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.1627    mrr@10 : 0.3658    ndcg@10 : 0.2017    hit@10 : 0.7103    precision@10 : 0.1491
  Test result:
  recall@10 : 0.1821    mrr@10 : 0.4205    ndcg@10 : 0.2443    hit@10 : 0.733    precision@10 : 0.1816
  
  build_graph_separately:True, learning_rate:0.001, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.1629    mrr@10 : 0.3603    ndcg@10 : 0.2003    hit@10 : 0.7119    precision@10 : 0.1495
  Test result:
  recall@10 : 0.1808    mrr@10 : 0.419    ndcg@10 : 0.2437    hit@10 : 0.7278    precision@10 : 0.1815
  
  build_graph_separately:True, learning_rate:0.0001, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.1506    mrr@10 : 0.3483    ndcg@10 : 0.1908    hit@10 : 0.6882    precision@10 : 0.1441
  Test result:
  recall@10 : 0.167    mrr@10 : 0.4085    ndcg@10 : 0.2343    hit@10 : 0.7033    precision@10 : 0.1756
  
  build_graph_separately:True, learning_rate:0.005, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.1532    mrr@10 : 0.3454    ndcg@10 : 0.1903    hit@10 : 0.6935    precision@10 : 0.1438
  Test result:
  recall@10 : 0.1708    mrr@10 : 0.3881    ndcg@10 : 0.2202    hit@10 : 0.7127    precision@10 : 0.1633
  
  build_graph_separately:False, learning_rate:0.001, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.1622    mrr@10 : 0.3651    ndcg@10 : 0.201    hit@10 : 0.7147    precision@10 : 0.15
  Test result:
  recall@10 : 0.1795    mrr@10 : 0.4227    ndcg@10 : 0.241    hit@10 : 0.73    precision@10 : 0.1781
  
  build_graph_separately:False, learning_rate:0.0001, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.1431    mrr@10 : 0.3377    ndcg@10 : 0.1832    hit@10 : 0.6698    precision@10 : 0.1389
  Test result:
  recall@10 : 0.1582    mrr@10 : 0.4031    ndcg@10 : 0.226    hit@10 : 0.6856    precision@10 : 0.1688
  
  build_graph_separately:True, learning_rate:0.0001, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.16    mrr@10 : 0.362    ndcg@10 : 0.2002    hit@10 : 0.7053    precision@10 : 0.1497
  Test result:
  recall@10 : 0.1777    mrr@10 : 0.4245    ndcg@10 : 0.2447    hit@10 : 0.7204    precision@10 : 0.1817
  
  build_graph_separately:False, learning_rate:0.005, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.1549    mrr@10 : 0.3427    ndcg@10 : 0.1893    hit@10 : 0.6998    precision@10 : 0.143
  Test result:
  recall@10 : 0.1673    mrr@10 : 0.3875    ndcg@10 : 0.2175    hit@10 : 0.7079    precision@10 : 0.1605
  
  build_graph_separately:True, learning_rate:0.001, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.161    mrr@10 : 0.3639    ndcg@10 : 0.2002    hit@10 : 0.7104    precision@10 : 0.148
  Test result:
  recall@10 : 0.1792    mrr@10 : 0.4236    ndcg@10 : 0.2442    hit@10 : 0.7253    precision@10 : 0.1817
  
  build_graph_separately:False, learning_rate:0.0001, loss_type:BPR, mess_dropout_rate:0.0, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.1408    mrr@10 : 0.3381    ndcg@10 : 0.1817    hit@10 : 0.6677    precision@10 : 0.1372
  Test result:
  recall@10 : 0.1553    mrr@10 : 0.3999    ndcg@10 : 0.2236    hit@10 : 0.6786    precision@10 : 0.1674
  
  build_graph_separately:False, learning_rate:0.001, loss_type:BPR, mess_dropout_rate:0.1, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.1627    mrr@10 : 0.3624    ndcg@10 : 0.201    hit@10 : 0.7122    precision@10 : 0.1497
  Test result:
  recall@10 : 0.1806    mrr@10 : 0.4206    ndcg@10 : 0.2431    hit@10 : 0.7308    precision@10 : 0.181
  ```


- **Logging Result**:

  ```yaml
  100%|██████████| 36/36 [437:01:04<00:00, 43701.79s/trial, best loss: -0.2019]
  best params:  {'build_graph_separately': True, 'learning_rate': 0.001, 'loss_type': 'BPR', 'mess_dropout_rate': 0.1, 'node_dropout_rate': 0.5}
  best result: 
  {'model': 'MCCLK', 'best_valid_score': 0.2019, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1634), ('mrr@10', 0.3644), ('ndcg@10', 0.2019), ('hit@10', 0.7127), ('precision@10', 0.1491)]), 'test_result': OrderedDict([('recall@10', 0.1832), ('mrr@10', 0.4242), ('ndcg@10', 0.2465), ('hit@10', 0.7341), ('precision@10', 0.1829)])}
  ```
  
  
