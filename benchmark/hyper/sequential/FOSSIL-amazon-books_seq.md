# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [FOSSIL](https://recbole.io/docs/user_guide/model/sequential/fossil.html)

- **Time cost**: 1835.97s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0005, 0.001, 0.003]
  reg_weight choice [1e-5, 1e-4]
  order_len choice [1, 2, 3]
  alpha choice [0.2, 0.4, 0.6]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  reg_weight: 1e-05
  order_len: 2
  alpha: 0.4
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  alpha:0.2, learning_rate:0.001, order_len:1, reg_weight:1e-05
  Valid result:
  recall@10 : 0.216    mrr@10 : 0.069    ndcg@10 : 0.1036    hit@10 : 0.216    precision@10 : 0.0216
  Test result:
  recall@10 : 0.1879    mrr@10 : 0.0632    ndcg@10 : 0.0925    hit@10 : 0.1879    precision@10 : 0.0188

  alpha:0.4, learning_rate:0.003, order_len:3, reg_weight:1e-05
  Valid result:
  recall@10 : 0.2067    mrr@10 : 0.0644    ndcg@10 : 0.0976    hit@10 : 0.2067    precision@10 : 0.0207
  Test result:
  recall@10 : 0.1754    mrr@10 : 0.0572    ndcg@10 : 0.0848    hit@10 : 0.1754    precision@10 : 0.0175

  alpha:0.4, learning_rate:0.003, order_len:3, reg_weight:0.0001
  Valid result:
  recall@10 : 0.2061    mrr@10 : 0.0644    ndcg@10 : 0.0975    hit@10 : 0.2061    precision@10 : 0.0206
  Test result:
  recall@10 : 0.1752    mrr@10 : 0.0572    ndcg@10 : 0.0848    hit@10 : 0.1752    precision@10 : 0.0175

  alpha:0.2, learning_rate:0.003, order_len:3, reg_weight:1e-05
  Valid result:
  recall@10 : 0.2081    mrr@10 : 0.0648    ndcg@10 : 0.0983    hit@10 : 0.2081    precision@10 : 0.0208
  Test result:
  recall@10 : 0.1801    mrr@10 : 0.0597    ndcg@10 : 0.0879    hit@10 : 0.1801    precision@10 : 0.018

  alpha:0.2, learning_rate:0.0005, order_len:2, reg_weight:0.0001
  Valid result:
  recall@10 : 0.2207    mrr@10 : 0.0645    ndcg@10 : 0.1011    hit@10 : 0.2207    precision@10 : 0.0221
  Test result:
  recall@10 : 0.1923    mrr@10 : 0.0599    ndcg@10 : 0.0909    hit@10 : 0.1923    precision@10 : 0.0192

  alpha:0.2, learning_rate:0.003, order_len:3, reg_weight:0.0001
  Valid result:
  recall@10 : 0.2054    mrr@10 : 0.0654    ndcg@10 : 0.0981    hit@10 : 0.2054    precision@10 : 0.0205
  Test result:
  recall@10 : 0.1768    mrr@10 : 0.0587    ndcg@10 : 0.0863    hit@10 : 0.1768    precision@10 : 0.0177

  alpha:0.2, learning_rate:0.001, order_len:2, reg_weight:0.0001
  Valid result:
  recall@10 : 0.2231    mrr@10 : 0.0684    ndcg@10 : 0.1047    hit@10 : 0.2231    precision@10 : 0.0223
  Test result:
  recall@10 : 0.1943    mrr@10 : 0.0616    ndcg@10 : 0.0927    hit@10 : 0.1943    precision@10 : 0.0194

  alpha:0.2, learning_rate:0.003, order_len:2, reg_weight:0.0001
  Valid result:
  recall@10 : 0.2118    mrr@10 : 0.0675    ndcg@10 : 0.1013    hit@10 : 0.2118    precision@10 : 0.0212
  Test result:
  recall@10 : 0.1839    mrr@10 : 0.0604    ndcg@10 : 0.0893    hit@10 : 0.1839    precision@10 : 0.0184

  alpha:0.4, learning_rate:0.001, order_len:2, reg_weight:1e-05
  Valid result:
  recall@10 : 0.2236    mrr@10 : 0.0683    ndcg@10 : 0.1048    hit@10 : 0.2236    precision@10 : 0.0224
  Test result:
  recall@10 : 0.1906    mrr@10 : 0.0617    ndcg@10 : 0.092    hit@10 : 0.1906    precision@10 : 0.0191

  alpha:0.6, learning_rate:0.003, order_len:3, reg_weight:1e-05
  Valid result:
  recall@10 : 0.2046    mrr@10 : 0.0633    ndcg@10 : 0.0963    hit@10 : 0.2046    precision@10 : 0.0205
  Test result:
  recall@10 : 0.1732    mrr@10 : 0.0554    ndcg@10 : 0.083    hit@10 : 0.1732    precision@10 : 0.0173

  alpha:0.6, learning_rate:0.003, order_len:3, reg_weight:0.0001
  Valid result:
  recall@10 : 0.2044    mrr@10 : 0.0631    ndcg@10 : 0.0961    hit@10 : 0.2044    precision@10 : 0.0204
  Test result:
  recall@10 : 0.1728    mrr@10 : 0.0554    ndcg@10 : 0.0829    hit@10 : 0.1728    precision@10 : 0.0173

  alpha:0.4, learning_rate:0.0005, order_len:1, reg_weight:1e-05
  Valid result:
  recall@10 : 0.2168    mrr@10 : 0.0661    ndcg@10 : 0.1015    hit@10 : 0.2168    precision@10 : 0.0217
  Test result:
  recall@10 : 0.1878    mrr@10 : 0.0592    ndcg@10 : 0.0894    hit@10 : 0.1878    precision@10 : 0.0188

  alpha:0.4, learning_rate:0.0005, order_len:2, reg_weight:1e-05
  Valid result:
  recall@10 : 0.2233    mrr@10 : 0.0654    ndcg@10 : 0.1025    hit@10 : 0.2233    precision@10 : 0.0223
  Test result:
  recall@10 : 0.1926    mrr@10 : 0.0591    ndcg@10 : 0.0904    hit@10 : 0.1926    precision@10 : 0.0193

  alpha:0.6, learning_rate:0.0005, order_len:2, reg_weight:1e-05
  Valid result:
  recall@10 : 0.2223    mrr@10 : 0.0656    ndcg@10 : 0.1023    hit@10 : 0.2223    precision@10 : 0.0222
  Test result:
  recall@10 : 0.1871    mrr@10 : 0.0578    ndcg@10 : 0.0881    hit@10 : 0.1871    precision@10 : 0.0187

  alpha:0.2, learning_rate:0.0005, order_len:3, reg_weight:0.0001
  Valid result:
  recall@10 : 0.2188    mrr@10 : 0.0632    ndcg@10 : 0.0995    hit@10 : 0.2188    precision@10 : 0.0219
  Test result:
  recall@10 : 0.1883    mrr@10 : 0.0566    ndcg@10 : 0.0873    hit@10 : 0.1883    precision@10 : 0.0188

  alpha:0.6, learning_rate:0.003, order_len:1, reg_weight:0.0001
  Valid result:
  recall@10 : 0.2031    mrr@10 : 0.062    ndcg@10 : 0.095    hit@10 : 0.2031    precision@10 : 0.0203
  Test result:
  recall@10 : 0.1727    mrr@10 : 0.0544    ndcg@10 : 0.0819    hit@10 : 0.1727    precision@10 : 0.0173

  alpha:0.4, learning_rate:0.001, order_len:3, reg_weight:0.0001
  Valid result:
  recall@10 : 0.2183    mrr@10 : 0.0657    ndcg@10 : 0.1014    hit@10 : 0.2183    precision@10 : 0.0218
  Test result:
  recall@10 : 0.187    mrr@10 : 0.059    ndcg@10 : 0.0889    hit@10 : 0.187    precision@10 : 0.0187

  alpha:0.4, learning_rate:0.0005, order_len:1, reg_weight:0.0001
  Valid result:
  recall@10 : 0.2158    mrr@10 : 0.0658    ndcg@10 : 0.101    hit@10 : 0.2158    precision@10 : 0.0216
  Test result:
  recall@10 : 0.1876    mrr@10 : 0.0591    ndcg@10 : 0.0892    hit@10 : 0.1876    precision@10 : 0.0188

  alpha:0.4, learning_rate:0.003, order_len:2, reg_weight:1e-05
  Valid result:
  recall@10 : 0.2096    mrr@10 : 0.0668    ndcg@10 : 0.1003    hit@10 : 0.2096    precision@10 : 0.021
  Test result:
  recall@10 : 0.1814    mrr@10 : 0.0601    ndcg@10 : 0.0885    hit@10 : 0.1814    precision@10 : 0.0181
  ```

- **Logging Result**:

  ```yaml
  INFO  Early stop triggered. Stopping iterations as condition is reach.
  35%|███▌      | 19/54 [9:41:23<17:50:58, 1835.97s/trial, best loss: -0.1048]
  best params:  {'alpha': 0.4, 'learning_rate': 0.001, 'order_len': 2, 'reg_weight': 1e-05}
  best result: 
  {'model': 'FOSSIL', 'best_valid_score': 0.1048, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2236), ('mrr@10', 0.0683), ('ndcg@10', 0.1048), ('hit@10', 0.2236), ('precision@10', 0.0224)]), 'test_result': OrderedDict([('recall@10', 0.1906), ('mrr@10', 0.0617), ('ndcg@10', 0.092), ('hit@10', 0.1906), ('precision@10', 0.0191)])}
  ```
