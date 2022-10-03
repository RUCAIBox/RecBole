# General Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_general.md)

- **Model**: [DGCF](https://recbole.io/docs/user_guide/model/general/dgcf.html)

- **Time cost**: 93762.00s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [5e-4,1e-3, 5e-3,1e-2]                     
  n_factors choice [2, 4, 8]                                                  
  reg_weight choice [1e-3,1e-2]                                         
  cor_weight choice [1e-3,1e-2]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  n_factors: 2
  reg_weight: 0.001
  cor_weight: 0.01
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  cor_weight:0.001, learning_rate:0.005, n_factors:4, reg_weight:0.01
  Valid result:
  recall@10 : 0.1421    mrr@10 : 0.1008    ndcg@10 : 0.0922    hit@10 : 0.2316    precision@10 : 0.0277
  Test result:
  recall@10 : 0.1427    mrr@10 : 0.1055    ndcg@10 : 0.0957    hit@10 : 0.2311    precision@10 : 0.0284

  cor_weight:0.001, learning_rate:0.005, n_factors:8, reg_weight:0.01
  Valid result:
  recall@10 : 0.1407    mrr@10 : 0.0996    ndcg@10 : 0.091    hit@10 : 0.229    precision@10 : 0.0276
  Test result:
  recall@10 : 0.1421    mrr@10 : 0.1055    ndcg@10 : 0.0953    hit@10 : 0.2302    precision@10 : 0.0284

  cor_weight:0.001, learning_rate:0.0005, n_factors:2, reg_weight:0.001
  Valid result:
  recall@10 : 0.1601    mrr@10 : 0.114    ndcg@10 : 0.1044    hit@10 : 0.2577    precision@10 : 0.0316
  Test result:
  recall@10 : 0.1644    mrr@10 : 0.1224    ndcg@10 : 0.1106    hit@10 : 0.2626    precision@10 : 0.0332

  cor_weight:0.001, learning_rate:0.01, n_factors:4, reg_weight:0.001
  Valid result:
  recall@10 : 0.1201    mrr@10 : 0.0834    ndcg@10 : 0.0766    hit@10 : 0.1985    precision@10 : 0.0234
  Test result:
  recall@10 : 0.1214    mrr@10 : 0.0895    ndcg@10 : 0.0802    hit@10 : 0.2011    precision@10 : 0.0242

  cor_weight:0.01, learning_rate:0.001, n_factors:8, reg_weight:0.001
  Valid result:
  recall@10 : 0.1621    mrr@10 : 0.1156    ndcg@10 : 0.1058    hit@10 : 0.2608    precision@10 : 0.0321
  Test result:
  recall@10 : 0.1662    mrr@10 : 0.124    ndcg@10 : 0.1127    hit@10 : 0.2653    precision@10 : 0.0335

  cor_weight:0.001, learning_rate:0.0005, n_factors:2, reg_weight:0.01
  Valid result:
  recall@10 : 0.1595    mrr@10 : 0.1139    ndcg@10 : 0.1041    hit@10 : 0.2571    precision@10 : 0.0315
  Test result:
  recall@10 : 0.1645    mrr@10 : 0.1222    ndcg@10 : 0.1104    hit@10 : 0.2634    precision@10 : 0.0333

  cor_weight:0.001, learning_rate:0.0005, n_factors:4, reg_weight:0.001
  Valid result:
  recall@10 : 0.1586    mrr@10 : 0.1138    ndcg@10 : 0.104    hit@10 : 0.2558    precision@10 : 0.0315
  Test result:
  recall@10 : 0.1627    mrr@10 : 0.1222    ndcg@10 : 0.1104    hit@10 : 0.2601    precision@10 : 0.0329

  cor_weight:0.001, learning_rate:0.005, n_factors:8, reg_weight:0.001
  Valid result:
  recall@10 : 0.1404    mrr@10 : 0.0991    ndcg@10 : 0.0907    hit@10 : 0.2284    precision@10 : 0.0274
  Test result:
  recall@10 : 0.1416    mrr@10 : 0.104    ndcg@10 : 0.0941    hit@10 : 0.2294    precision@10 : 0.0282

  cor_weight:0.01, learning_rate:0.0005, n_factors:4, reg_weight:0.01
  Valid result:
  recall@10 : 0.1588    mrr@10 : 0.1133    ndcg@10 : 0.1034    hit@10 : 0.2552    precision@10 : 0.0314
  Test result:
  recall@10 : 0.1657    mrr@10 : 0.1236    ndcg@10 : 0.1119    hit@10 : 0.2634    precision@10 : 0.0334

  cor_weight:0.01, learning_rate:0.0005, n_factors:8, reg_weight:0.001
  Valid result:
  recall@10 : 0.1557    mrr@10 : 0.1119    ndcg@10 : 0.1018    hit@10 : 0.2523    precision@10 : 0.0311
  Test result:
  recall@10 : 0.1602    mrr@10 : 0.1201    ndcg@10 : 0.1084    hit@10 : 0.257    precision@10 : 0.0325

  cor_weight:0.001, learning_rate:0.0005, n_factors:8, reg_weight:0.001
  Valid result:
  recall@10 : 0.1574    mrr@10 : 0.114    ndcg@10 : 0.1035    hit@10 : 0.2548    precision@10 : 0.0313
  Test result:
  recall@10 : 0.1613    mrr@10 : 0.1204    ndcg@10 : 0.1089    hit@10 : 0.2591    precision@10 : 0.0328

  cor_weight:0.01, learning_rate:0.005, n_factors:4, reg_weight:0.001
  Valid result:
  recall@10 : 0.1426    mrr@10 : 0.1007    ndcg@10 : 0.0924    hit@10 : 0.2305    precision@10 : 0.0279
  Test result:
  recall@10 : 0.1424    mrr@10 : 0.1066    ndcg@10 : 0.096    hit@10 : 0.231    precision@10 : 0.0284

  cor_weight:0.001, learning_rate:0.001, n_factors:2, reg_weight:0.01
  Valid result:
  recall@10 : 0.1658    mrr@10 : 0.1192    ndcg@10 : 0.1087    hit@10 : 0.2661    precision@10 : 0.0326
  Test result:
  recall@10 : 0.1699    mrr@10 : 0.1268    ndcg@10 : 0.1149    hit@10 : 0.2697    precision@10 : 0.0342

  cor_weight:0.01, learning_rate:0.005, n_factors:2, reg_weight:0.001
  Valid result:
  recall@10 : 0.1435    mrr@10 : 0.103    ndcg@10 : 0.0936    hit@10 : 0.2339    precision@10 : 0.0282
  Test result:
  recall@10 : 0.1451    mrr@10 : 0.108    ndcg@10 : 0.0971    hit@10 : 0.236    precision@10 : 0.0291

  cor_weight:0.01, learning_rate:0.0005, n_factors:4, reg_weight:0.001
  Valid result:
  recall@10 : 0.1583    mrr@10 : 0.1135    ndcg@10 : 0.1039    hit@10 : 0.255    precision@10 : 0.0315
  Test result:
  recall@10 : 0.1623    mrr@10 : 0.1215    ndcg@10 : 0.1099    hit@10 : 0.2593    precision@10 : 0.0331

  cor_weight:0.01, learning_rate:0.0005, n_factors:8, reg_weight:0.01
  Valid result:
  recall@10 : 0.1569    mrr@10 : 0.1115    ndcg@10 : 0.1019    hit@10 : 0.2537    precision@10 : 0.0311
  Test result:
  recall@10 : 0.1599    mrr@10 : 0.1207    ndcg@10 : 0.1084    hit@10 : 0.2555    precision@10 : 0.0324

  cor_weight:0.01, learning_rate:0.005, n_factors:4, reg_weight:0.01
  Valid result:
  recall@10 : 0.1417    mrr@10 : 0.1018    ndcg@10 : 0.0928    hit@10 : 0.2303    precision@10 : 0.0277
  Test result:
  recall@10 : 0.1439    mrr@10 : 0.1071    ndcg@10 : 0.0965    hit@10 : 0.2332    precision@10 : 0.0287

  cor_weight:0.001, learning_rate:0.005, n_factors:2, reg_weight:0.01
  Valid result:
  recall@10 : 0.1421    mrr@10 : 0.101    ndcg@10 : 0.0922    hit@10 : 0.2301    precision@10 : 0.0278
  Test result:
  recall@10 : 0.1439    mrr@10 : 0.1078    ndcg@10 : 0.0969    hit@10 : 0.2339    precision@10 : 0.0287

  cor_weight:0.01, learning_rate:0.005, n_factors:2, reg_weight:0.01
  Valid result:
  recall@10 : 0.1429    mrr@10 : 0.1023    ndcg@10 : 0.0932    hit@10 : 0.2332    precision@10 : 0.0283
  Test result:
  recall@10 : 0.1457    mrr@10 : 0.1071    ndcg@10 : 0.0971    hit@10 : 0.2377    precision@10 : 0.0292

  cor_weight:0.001, learning_rate:0.001, n_factors:2, reg_weight:0.001
  Valid result:
  recall@10 : 0.1651    mrr@10 : 0.1186    ndcg@10 : 0.1084    hit@10 : 0.2645    precision@10 : 0.0325
  Test result:
  recall@10 : 0.1692    mrr@10 : 0.1259    ndcg@10 : 0.1144    hit@10 : 0.268    precision@10 : 0.0339

  cor_weight:0.01, learning_rate:0.01, n_factors:8, reg_weight:0.001
  Valid result:
  recall@10 : 0.1191    mrr@10 : 0.0847    ndcg@10 : 0.0766    hit@10 : 0.1971    precision@10 : 0.0234
  Test result:
  recall@10 : 0.1218    mrr@10 : 0.0894    ndcg@10 : 0.08    hit@10 : 0.203    precision@10 : 0.0245

  cor_weight:0.001, learning_rate:0.0005, n_factors:4, reg_weight:0.01
  Valid result:
  recall@10 : 0.1593    mrr@10 : 0.1134    ndcg@10 : 0.1039    hit@10 : 0.2574    precision@10 : 0.0316
  Test result:
  recall@10 : 0.1636    mrr@10 : 0.1227    ndcg@10 : 0.1107    hit@10 : 0.2611    precision@10 : 0.0332

  cor_weight:0.01, learning_rate:0.001, n_factors:2, reg_weight:0.001
  Valid result:
  recall@10 : 0.1656    mrr@10 : 0.1187    ndcg@10 : 0.1088    hit@10 : 0.2649    precision@10 : 0.0327
  Test result:
  recall@10 : 0.169    mrr@10 : 0.1273    ndcg@10 : 0.115    hit@10 : 0.2689    precision@10 : 0.0341

  cor_weight:0.01, learning_rate:0.01, n_factors:2, reg_weight:0.001
  Valid result:
  recall@10 : 0.1208    mrr@10 : 0.0853    ndcg@10 : 0.0779    hit@10 : 0.1998    precision@10 : 0.0237
  Test result:
  recall@10 : 0.1234    mrr@10 : 0.0914    ndcg@10 : 0.0818    hit@10 : 0.2041    precision@10 : 0.0245

  cor_weight:0.001, learning_rate:0.005, n_factors:4, reg_weight:0.001
  Valid result:
  recall@10 : 0.1408    mrr@10 : 0.1006    ndcg@10 : 0.0918    hit@10 : 0.2311    precision@10 : 0.0279
  Test result:
  recall@10 : 0.1437    mrr@10 : 0.1061    ndcg@10 : 0.096    hit@10 : 0.2313    precision@10 : 0.0284

  cor_weight:0.001, learning_rate:0.0005, n_factors:8, reg_weight:0.01
  Valid result:
  recall@10 : 0.1574    mrr@10 : 0.1142    ndcg@10 : 0.1036    hit@10 : 0.2547    precision@10 : 0.0313
  Test result:
  recall@10 : 0.1616    mrr@10 : 0.1201    ndcg@10 : 0.1088    hit@10 : 0.2584    precision@10 : 0.0327

  cor_weight:0.01, learning_rate:0.001, n_factors:4, reg_weight:0.01
  Valid result:
  recall@10 : 0.161    mrr@10 : 0.1159    ndcg@10 : 0.1056    hit@10 : 0.2584    precision@10 : 0.0317
  Test result:
  recall@10 : 0.1664    mrr@10 : 0.1238    ndcg@10 : 0.112    hit@10 : 0.265    precision@10 : 0.0335

  cor_weight:0.001, learning_rate:0.001, n_factors:8, reg_weight:0.01
  Valid result:
  recall@10 : 0.1611    mrr@10 : 0.1153    ndcg@10 : 0.1052    hit@10 : 0.259    precision@10 : 0.0319
  Test result:
  recall@10 : 0.1662    mrr@10 : 0.1237    ndcg@10 : 0.1119    hit@10 : 0.2637    precision@10 : 0.0334

  cor_weight:0.001, learning_rate:0.01, n_factors:2, reg_weight:0.01
  Valid result:
  recall@10 : 0.12    mrr@10 : 0.0852    ndcg@10 : 0.0778    hit@10 : 0.1975    precision@10 : 0.0234
  Test result:
  recall@10 : 0.1229    mrr@10 : 0.0913    ndcg@10 : 0.0815    hit@10 : 0.2041    precision@10 : 0.0245

  cor_weight:0.001, learning_rate:0.001, n_factors:4, reg_weight:0.01
  Valid result:
  recall@10 : 0.1654    mrr@10 : 0.1177    ndcg@10 : 0.1079    hit@10 : 0.265    precision@10 : 0.0325
  Test result:
  recall@10 : 0.1684    mrr@10 : 0.1267    ndcg@10 : 0.1144    hit@10 : 0.268    precision@10 : 0.0339

  cor_weight:0.001, learning_rate:0.001, n_factors:4, reg_weight:0.001
  Valid result:
  recall@10 : 0.1621    mrr@10 : 0.1165    ndcg@10 : 0.1062    hit@10 : 0.2604    precision@10 : 0.032
  Test result:
  recall@10 : 0.1661    mrr@10 : 0.1245    ndcg@10 : 0.1126    hit@10 : 0.265    precision@10 : 0.0335

  cor_weight:0.01, learning_rate:0.01, n_factors:2, reg_weight:0.01
  Valid result:
  recall@10 : 0.1217    mrr@10 : 0.0856    ndcg@10 : 0.0784    hit@10 : 0.1995    precision@10 : 0.0236
  Test result:
  recall@10 : 0.1228    mrr@10 : 0.0912    ndcg@10 : 0.0818    hit@10 : 0.2041    precision@10 : 0.0246

  cor_weight:0.001, learning_rate:0.01, n_factors:2, reg_weight:0.001
  Valid result:
  recall@10 : 0.1203    mrr@10 : 0.0855    ndcg@10 : 0.0778    hit@10 : 0.1993    precision@10 : 0.0235
  Test result:
  recall@10 : 0.1227    mrr@10 : 0.092    ndcg@10 : 0.0816    hit@10 : 0.2045    precision@10 : 0.0245
  ```

- **Logging Result**:

  ```yaml
  INFO  Early stop triggered. Stopping iterations as condition is reach.
  69%|██████▉   | 33/48 [859:29:05<390:40:29, 93762.00s/trial, best loss: -0.1088]
  best params:  {'cor_weight': 0.01, 'learning_rate': 0.001, 'n_factors': 2, 'reg_weight': 0.001}
  best result: 
  {'model': 'DGCF', 'best_valid_score': 0.1088, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1656), ('mrr@10', 0.1187), ('ndcg@10', 0.1088), ('hit@10', 0.2649), ('precision@10', 0.0327)]), 'test_result': OrderedDict([('recall@10', 0.169), ('mrr@10', 0.1273), ('ndcg@10', 0.115), ('hit@10', 0.2689), ('precision@10', 0.0341)])}
  ```
