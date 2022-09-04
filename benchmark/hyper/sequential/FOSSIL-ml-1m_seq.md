# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [FOSSIL](https://recbole.io/docs/user_guide/model/sequential/fossil.html)

- **Time cost**: 1640.69s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0005, 0.001, 0.003]
  reg_weight choice [1e-5, 1e-4]
  order_len choice [1, 2, 3]
  alpha choice [0.2, 0.4, 0.6]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.003
  reg_weight: 0.0001
  order_len: 1
  alpha: 0.6
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  alpha:0.2, learning_rate:0.0005, order_len:1, reg_weight:1e-05
  Valid result:
  recall@10 : 0.0406    mrr@10 : 0.0099    ndcg@10 : 0.0169    hit@10 : 0.0406    precision@10 : 0.0041
  Test result:
  recall@10 : 0.0416    mrr@10 : 0.0101    ndcg@10 : 0.0172    hit@10 : 0.0416    precision@10 : 0.0042

  alpha:0.2, learning_rate:0.003, order_len:3, reg_weight:1e-05
  Valid result:
  recall@10 : 0.1404    mrr@10 : 0.0316    ndcg@10 : 0.0563    hit@10 : 0.1404    precision@10 : 0.014
  Test result:
  recall@10 : 0.1334    mrr@10 : 0.0313    ndcg@10 : 0.0547    hit@10 : 0.1334    precision@10 : 0.0133

  alpha:0.4, learning_rate:0.001, order_len:3, reg_weight:1e-05
  Valid result:
  recall@10 : 0.1404    mrr@10 : 0.0306    ndcg@10 : 0.0556    hit@10 : 0.1404    precision@10 : 0.014
  Test result:
  recall@10 : 0.1386    mrr@10 : 0.0313    ndcg@10 : 0.0558    hit@10 : 0.1386    precision@10 : 0.0139

  alpha:0.2, learning_rate:0.0005, order_len:1, reg_weight:0.0001
  Valid result:
  recall@10 : 0.0391    mrr@10 : 0.0101    ndcg@10 : 0.0167    hit@10 : 0.0391    precision@10 : 0.0039
  Test result:
  recall@10 : 0.0411    mrr@10 : 0.0102    ndcg@10 : 0.0171    hit@10 : 0.0411    precision@10 : 0.0041

  alpha:0.4, learning_rate:0.001, order_len:2, reg_weight:1e-05
  Valid result:
  recall@10 : 0.1437    mrr@10 : 0.0324    ndcg@10 : 0.0578    hit@10 : 0.1437    precision@10 : 0.0144
  Test result:
  recall@10 : 0.1409    mrr@10 : 0.0319    ndcg@10 : 0.0568    hit@10 : 0.1409    precision@10 : 0.0141

  alpha:0.2, learning_rate:0.003, order_len:1, reg_weight:0.0001
  Valid result:
  recall@10 : 0.1425    mrr@10 : 0.0356    ndcg@10 : 0.0603    hit@10 : 0.1425    precision@10 : 0.0143
  Test result:
  recall@10 : 0.1314    mrr@10 : 0.0338    ndcg@10 : 0.0564    hit@10 : 0.1314    precision@10 : 0.0131

  alpha:0.4, learning_rate:0.003, order_len:3, reg_weight:0.0001
  Valid result:
  recall@10 : 0.1429    mrr@10 : 0.0326    ndcg@10 : 0.0577    hit@10 : 0.1429    precision@10 : 0.0143
  Test result:
  recall@10 : 0.143    mrr@10 : 0.0332    ndcg@10 : 0.0584    hit@10 : 0.143    precision@10 : 0.0143

  alpha:0.4, learning_rate:0.0005, order_len:3, reg_weight:0.0001
  Valid result:
  recall@10 : 0.1402    mrr@10 : 0.0304    ndcg@10 : 0.0554    hit@10 : 0.1402    precision@10 : 0.014
  Test result:
  recall@10 : 0.1363    mrr@10 : 0.0302    ndcg@10 : 0.0544    hit@10 : 0.1363    precision@10 : 0.0136

  alpha:0.2, learning_rate:0.0005, order_len:2, reg_weight:1e-05
  Valid result:
  recall@10 : 0.1251    mrr@10 : 0.0273    ndcg@10 : 0.0495    hit@10 : 0.1251    precision@10 : 0.0125
  Test result:
  recall@10 : 0.124    mrr@10 : 0.0277    ndcg@10 : 0.0497    hit@10 : 0.124    precision@10 : 0.0124

  alpha:0.4, learning_rate:0.003, order_len:1, reg_weight:0.0001
  Valid result:
  recall@10 : 0.1442    mrr@10 : 0.0367    ndcg@10 : 0.0615    hit@10 : 0.1442    precision@10 : 0.0144
  Test result:
  recall@10 : 0.1323    mrr@10 : 0.0344    ndcg@10 : 0.0572    hit@10 : 0.1323    precision@10 : 0.0132

  alpha:0.2, learning_rate:0.0005, order_len:3, reg_weight:1e-05
  Valid result:
  recall@10 : 0.1303    mrr@10 : 0.028    ndcg@10 : 0.0512    hit@10 : 0.1303    precision@10 : 0.013
  Test result:
  recall@10 : 0.129    mrr@10 : 0.0286    ndcg@10 : 0.0514    hit@10 : 0.129    precision@10 : 0.0129

  alpha:0.2, learning_rate:0.0005, order_len:3, reg_weight:0.0001
  Valid result:
  recall@10 : 0.1356    mrr@10 : 0.0289    ndcg@10 : 0.0531    hit@10 : 0.1356    precision@10 : 0.0136
  Test result:
  recall@10 : 0.1338    mrr@10 : 0.029    ndcg@10 : 0.0529    hit@10 : 0.1338    precision@10 : 0.0134

  alpha:0.6, learning_rate:0.001, order_len:1, reg_weight:0.0001
  Valid result:
  recall@10 : 0.151    mrr@10 : 0.0373    ndcg@10 : 0.0636    hit@10 : 0.151    precision@10 : 0.0151
  Test result:
  recall@10 : 0.1389    mrr@10 : 0.0352    ndcg@10 : 0.0593    hit@10 : 0.1389    precision@10 : 0.0139

  alpha:0.4, learning_rate:0.0005, order_len:2, reg_weight:0.0001
  Valid result:
  recall@10 : 0.1429    mrr@10 : 0.0308    ndcg@10 : 0.0563    hit@10 : 0.1429    precision@10 : 0.0143
  Test result:
  recall@10 : 0.1397    mrr@10 : 0.0316    ndcg@10 : 0.0564    hit@10 : 0.1397    precision@10 : 0.014

  alpha:0.6, learning_rate:0.001, order_len:3, reg_weight:0.0001
  Valid result:
  recall@10 : 0.148    mrr@10 : 0.0324    ndcg@10 : 0.0587    hit@10 : 0.148    precision@10 : 0.0148
  Test result:
  recall@10 : 0.143    mrr@10 : 0.0321    ndcg@10 : 0.0575    hit@10 : 0.143    precision@10 : 0.0143

  alpha:0.6, learning_rate:0.003, order_len:1, reg_weight:0.0001
  Valid result:
  recall@10 : 0.1522    mrr@10 : 0.0373    ndcg@10 : 0.0638    hit@10 : 0.1522    precision@10 : 0.0152
  Test result:
  recall@10 : 0.1346    mrr@10 : 0.0354    ndcg@10 : 0.0585    hit@10 : 0.1346    precision@10 : 0.0135

  alpha:0.4, learning_rate:0.003, order_len:2, reg_weight:0.0001
  Valid result:
  recall@10 : 0.1437    mrr@10 : 0.0333    ndcg@10 : 0.0585    hit@10 : 0.1437    precision@10 : 0.0144
  Test result:
  recall@10 : 0.1442    mrr@10 : 0.0346    ndcg@10 : 0.0598    hit@10 : 0.1442    precision@10 : 0.0144

  alpha:0.6, learning_rate:0.003, order_len:3, reg_weight:0.0001
  Valid result:
  recall@10 : 0.1492    mrr@10 : 0.0339    ndcg@10 : 0.0602    hit@10 : 0.1492    precision@10 : 0.0149
  Test result:
  recall@10 : 0.1464    mrr@10 : 0.0345    ndcg@10 : 0.0602    hit@10 : 0.1464    precision@10 : 0.0146

  alpha:0.6, learning_rate:0.001, order_len:2, reg_weight:0.0001
  Valid result:
  recall@10 : 0.1474    mrr@10 : 0.0326    ndcg@10 : 0.0587    hit@10 : 0.1474    precision@10 : 0.0147
  Test result:
  recall@10 : 0.1442    mrr@10 : 0.033    ndcg@10 : 0.0585    hit@10 : 0.1442    precision@10 : 0.0144

  alpha:0.4, learning_rate:0.0005, order_len:2, reg_weight:1e-05
  Valid result:
  recall@10 : 0.1346    mrr@10 : 0.0291    ndcg@10 : 0.0531    hit@10 : 0.1346    precision@10 : 0.0135
  Test result:
  recall@10 : 0.1295    mrr@10 : 0.0291    ndcg@10 : 0.052    hit@10 : 0.1295    precision@10 : 0.0129

  alpha:0.4, learning_rate:0.003, order_len:1, reg_weight:1e-05
  Valid result:
  recall@10 : 0.1513    mrr@10 : 0.0375    ndcg@10 : 0.0638    hit@10 : 0.1513    precision@10 : 0.0151
  Test result:
  recall@10 : 0.1379    mrr@10 : 0.0356    ndcg@10 : 0.0593    hit@10 : 0.1379    precision@10 : 0.0138

  alpha:0.2, learning_rate:0.001, order_len:3, reg_weight:0.0001
  Valid result:
  recall@10 : 0.1384    mrr@10 : 0.029    ndcg@10 : 0.0538    hit@10 : 0.1384    precision@10 : 0.0138
  Test result:
  recall@10 : 0.1338    mrr@10 : 0.0291    ndcg@10 : 0.0529    hit@10 : 0.1338    precision@10 : 0.0134

  alpha:0.6, learning_rate:0.001, order_len:3, reg_weight:1e-05
  Valid result:
  recall@10 : 0.148    mrr@10 : 0.0324    ndcg@10 : 0.0586    hit@10 : 0.148    precision@10 : 0.0148
  Test result:
  recall@10 : 0.1435    mrr@10 : 0.0326    ndcg@10 : 0.0579    hit@10 : 0.1435    precision@10 : 0.0144

  alpha:0.4, learning_rate:0.0005, order_len:1, reg_weight:0.0001
  Valid result:
  recall@10 : 0.1435    mrr@10 : 0.0344    ndcg@10 : 0.0596    hit@10 : 0.1435    precision@10 : 0.0144
  Test result:
  recall@10 : 0.1314    mrr@10 : 0.0336    ndcg@10 : 0.0563    hit@10 : 0.1314    precision@10 : 0.0131

  alpha:0.6, learning_rate:0.003, order_len:3, reg_weight:1e-05
  Valid result:
  recall@10 : 0.1505    mrr@10 : 0.0345    ndcg@10 : 0.0609    hit@10 : 0.1505    precision@10 : 0.0151
  Test result:
  recall@10 : 0.1445    mrr@10 : 0.0334    ndcg@10 : 0.0589    hit@10 : 0.1445    precision@10 : 0.0145

  alpha:0.4, learning_rate:0.001, order_len:2, reg_weight:0.0001
  Valid result:
  recall@10 : 0.1459    mrr@10 : 0.0318    ndcg@10 : 0.0578    hit@10 : 0.1459    precision@10 : 0.0146
  Test result:
  recall@10 : 0.1401    mrr@10 : 0.032    ndcg@10 : 0.0567    hit@10 : 0.1401    precision@10 : 0.014
  ```

- **Logging Result**:

  ```yaml
  48%|████▊     | 26/54 [11:50:58<12:45:39, 1640.69s/trial, best loss: -0.0638]
  best params:  {'alpha': 0.6, 'learning_rate': 0.003, 'order_len': 1, 'reg_weight': 0.0001}
  best result: 
  {'model': 'FOSSIL', 'best_valid_score': 0.0638, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1522), ('mrr@10', 0.0373), ('ndcg@10', 0.0638), ('hit@10', 0.1522), ('precision@10', 0.0152)]), 'test_result': OrderedDict([('recall@10', 0.1346), ('mrr@10', 0.0354), ('ndcg@10', 0.0585), ('hit@10', 0.1346), ('precision@10', 0.0135)])}
  ```
