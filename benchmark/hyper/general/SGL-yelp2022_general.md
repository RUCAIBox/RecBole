# General Recommendation

- **Dataset**: [Yelp2022](../../md/yelp_general.md)

- **Model**: [SGL](https://recbole.io/docs/user_guide/model/general/sgl.html)

- **Time cost**: 19684.00s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [1e-4,1e-3,5e-3]
  ssl_tau choice [0.2,0.5]
  drop_ratio choice [0.1,0.2]
  ssl_weight choice [0.05,0.1]
  ```
  
- **Best parameters**:

  ```yaml
  learning_rate: 1e-3
  ssl_tau: 0.2
  drop_ratio: 0.2
  ssl_weight: 0.1
  ```
  
- **Hyper-parameter logging**:

  ```yaml
  drop_ratio:0.1, learning_rate:0.0001, ssl_tau:0.2, ssl_weight:0.05
  Valid result:
  recall@10 : 0.0295    mrr@10 : 0.016    ndcg@10 : 0.0159    hit@10 : 0.054    precision@10 : 0.006
  Test result:
  recall@10 : 0.0294    mrr@10 : 0.0158    ndcg@10 : 0.0159    hit@10 : 0.0536    precision@10 : 0.0059
  
  drop_ratio:0.2, learning_rate:0.001, ssl_tau:0.5, ssl_weight:0.1
  Valid result:
  recall@10 : 0.0934    mrr@10 : 0.0698    ndcg@10 : 0.0598    hit@10 : 0.1695    precision@10 : 0.02
  Test result:
  recall@10 : 0.0948    mrr@10 : 0.0716    ndcg@10 : 0.0611    hit@10 : 0.1719    precision@10 : 0.0205
  
  drop_ratio:0.1, learning_rate:0.005, ssl_tau:0.2, ssl_weight:0.1
  Valid result:
  recall@10 : 0.0989    mrr@10 : 0.0752    ndcg@10 : 0.0645    hit@10 : 0.1771    precision@10 : 0.0209
  Test result:
  recall@10 : 0.1023    mrr@10 : 0.0777    ndcg@10 : 0.0666    hit@10 : 0.1816    precision@10 : 0.0216
  
  drop_ratio:0.2, learning_rate:0.0001, ssl_tau:0.2, ssl_weight:0.1
  Valid result:
  recall@10 : 0.0402    mrr@10 : 0.0222    ndcg@10 : 0.0219    hit@10 : 0.0747    precision@10 : 0.0083
  Test result:
  recall@10 : 0.0407    mrr@10 : 0.0226    ndcg@10 : 0.0223    hit@10 : 0.0748    precision@10 : 0.0083
  
  drop_ratio:0.2, learning_rate:0.0001, ssl_tau:0.2, ssl_weight:0.05
  Valid result:
  recall@10 : 0.0343    mrr@10 : 0.0188    ndcg@10 : 0.0186    hit@10 : 0.0639    precision@10 : 0.0071
  Test result:
  recall@10 : 0.033    mrr@10 : 0.0188    ndcg@10 : 0.0182    hit@10 : 0.0626    precision@10 : 0.007
  
  drop_ratio:0.1, learning_rate:0.005, ssl_tau:0.5, ssl_weight:0.05
  Valid result:
  recall@10 : 0.0881    mrr@10 : 0.0656    ndcg@10 : 0.0567    hit@10 : 0.1593    precision@10 : 0.0185
  Test result:
  recall@10 : 0.089    mrr@10 : 0.0668    ndcg@10 : 0.0574    hit@10 : 0.1608    precision@10 : 0.0189
  
  drop_ratio:0.2, learning_rate:0.005, ssl_tau:0.2, ssl_weight:0.05
  Valid result:
  recall@10 : 0.0964    mrr@10 : 0.0731    ndcg@10 : 0.0628    hit@10 : 0.173    precision@10 : 0.0204
  Test result:
  recall@10 : 0.0968    mrr@10 : 0.0745    ndcg@10 : 0.0635    hit@10 : 0.1735    precision@10 : 0.0206
  
  drop_ratio:0.2, learning_rate:0.005, ssl_tau:0.5, ssl_weight:0.1
  Valid result:
  recall@10 : 0.0874    mrr@10 : 0.0655    ndcg@10 : 0.0563    hit@10 : 0.1585    precision@10 : 0.0183
  Test result:
  recall@10 : 0.0883    mrr@10 : 0.067    ndcg@10 : 0.0571    hit@10 : 0.1603    precision@10 : 0.0187
  
  drop_ratio:0.1, learning_rate:0.0001, ssl_tau:0.2, ssl_weight:0.1
  Valid result:
  recall@10 : 0.0337    mrr@10 : 0.0184    ndcg@10 : 0.0183    hit@10 : 0.0612    precision@10 : 0.0067
  Test result:
  recall@10 : 0.034    mrr@10 : 0.0186    ndcg@10 : 0.0186    hit@10 : 0.0599    precision@10 : 0.0066
  
  drop_ratio:0.2, learning_rate:0.0001, ssl_tau:0.5, ssl_weight:0.05
  Valid result:
  recall@10 : 0.0188    mrr@10 : 0.0103    ndcg@10 : 0.0101    hit@10 : 0.0357    precision@10 : 0.0038
  Test result:
  recall@10 : 0.0202    mrr@10 : 0.0106    ndcg@10 : 0.0107    hit@10 : 0.038    precision@10 : 0.0041
  
  drop_ratio:0.1, learning_rate:0.0001, ssl_tau:0.5, ssl_weight:0.05
  Valid result:
  recall@10 : 0.019    mrr@10 : 0.0098    ndcg@10 : 0.0099    hit@10 : 0.0352    precision@10 : 0.0037
  Test result:
  recall@10 : 0.0195    mrr@10 : 0.0098    ndcg@10 : 0.0101    hit@10 : 0.0355    precision@10 : 0.0038
  
  drop_ratio:0.2, learning_rate:0.0001, ssl_tau:0.5, ssl_weight:0.1
  Valid result:
  recall@10 : 0.0251    mrr@10 : 0.0131    ndcg@10 : 0.0132    hit@10 : 0.0472    precision@10 : 0.005
  Test result:
  recall@10 : 0.0259    mrr@10 : 0.0135    ndcg@10 : 0.0137    hit@10 : 0.0482    precision@10 : 0.0052
  
  drop_ratio:0.1, learning_rate:0.001, ssl_tau:0.5, ssl_weight:0.1
  Valid result:
  recall@10 : 0.0936    mrr@10 : 0.0702    ndcg@10 : 0.0601    hit@10 : 0.1701    precision@10 : 0.0201
  Test result:
  recall@10 : 0.0954    mrr@10 : 0.073    ndcg@10 : 0.0619    hit@10 : 0.1723    precision@10 : 0.0206
  
  drop_ratio:0.1, learning_rate:0.0001, ssl_tau:0.5, ssl_weight:0.1
  Valid result:
  recall@10 : 0.0248    mrr@10 : 0.0122    ndcg@10 : 0.0128    hit@10 : 0.0446    precision@10 : 0.0048
  Test result:
  recall@10 : 0.0252    mrr@10 : 0.0121    ndcg@10 : 0.0129    hit@10 : 0.0445    precision@10 : 0.0048
  
  drop_ratio:0.2, learning_rate:0.005, ssl_tau:0.5, ssl_weight:0.05
  Valid result:
  recall@10 : 0.086    mrr@10 : 0.0663    ndcg@10 : 0.0567    hit@10 : 0.156    precision@10 : 0.0182
  Test result:
  recall@10 : 0.0879    mrr@10 : 0.0674    ndcg@10 : 0.0575    hit@10 : 0.1588    precision@10 : 0.0186
  
  drop_ratio:0.1, learning_rate:0.001, ssl_tau:0.2, ssl_weight:0.05
  Valid result:
  recall@10 : 0.1021    mrr@10 : 0.079    ndcg@10 : 0.0672    hit@10 : 0.1827    precision@10 : 0.0218
  Test result:
  recall@10 : 0.1037    mrr@10 : 0.0809    ndcg@10 : 0.0687    hit@10 : 0.1856    precision@10 : 0.0224
  
  drop_ratio:0.1, learning_rate:0.001, ssl_tau:0.2, ssl_weight:0.1
  Valid result:
  recall@10 : 0.1012    mrr@10 : 0.0796    ndcg@10 : 0.0671    hit@10 : 0.1831    precision@10 : 0.0219
  Test result:
  recall@10 : 0.104    mrr@10 : 0.082    ndcg@10 : 0.069    hit@10 : 0.1867    precision@10 : 0.0225
  
  drop_ratio:0.1, learning_rate:0.001, ssl_tau:0.5, ssl_weight:0.05
  Valid result:
  recall@10 : 0.0959    mrr@10 : 0.0726    ndcg@10 : 0.0622    hit@10 : 0.1731    precision@10 : 0.0205
  Test result:
  recall@10 : 0.0978    mrr@10 : 0.0745    ndcg@10 : 0.0637    hit@10 : 0.1759    precision@10 : 0.0211
  
  drop_ratio:0.2, learning_rate:0.001, ssl_tau:0.2, ssl_weight:0.1
  Valid result:
  recall@10 : 0.1037    mrr@10 : 0.0791    ndcg@10 : 0.0674    hit@10 : 0.186    precision@10 : 0.0223
  Test result:
  recall@10 : 0.104    mrr@10 : 0.0806    ndcg@10 : 0.0684    hit@10 : 0.1866    precision@10 : 0.0226
  
  drop_ratio:0.1, learning_rate:0.005, ssl_tau:0.2, ssl_weight:0.05
  Valid result:
  recall@10 : 0.0979    mrr@10 : 0.0733    ndcg@10 : 0.0631    hit@10 : 0.1751    precision@10 : 0.0206
  Test result:
  recall@10 : 0.0994    mrr@10 : 0.0751    ndcg@10 : 0.0645    hit@10 : 0.1782    precision@10 : 0.0211
  
  drop_ratio:0.2, learning_rate:0.005, ssl_tau:0.2, ssl_weight:0.1
  Valid result:
  recall@10 : 0.0988    mrr@10 : 0.0745    ndcg@10 : 0.064    hit@10 : 0.1769    precision@10 : 0.0209
  Test result:
  recall@10 : 0.1001    mrr@10 : 0.076    ndcg@10 : 0.0652    hit@10 : 0.1782    precision@10 : 0.0212
  
  drop_ratio:0.2, learning_rate:0.001, ssl_tau:0.2, ssl_weight:0.05
  Valid result:
  recall@10 : 0.1002    mrr@10 : 0.0781    ndcg@10 : 0.0664    hit@10 : 0.1802    precision@10 : 0.0216
  Test result:
  recall@10 : 0.1027    mrr@10 : 0.0805    ndcg@10 : 0.0682    hit@10 : 0.1843    precision@10 : 0.0222
  
  drop_ratio:0.1, learning_rate:0.005, ssl_tau:0.5, ssl_weight:0.1
  Valid result:
  recall@10 : 0.0884    mrr@10 : 0.0658    ndcg@10 : 0.0568    hit@10 : 0.1599    precision@10 : 0.0185
  Test result:
  recall@10 : 0.0893    mrr@10 : 0.067    ndcg@10 : 0.0575    hit@10 : 0.1618    precision@10 : 0.019
  
  drop_ratio:0.2, learning_rate:0.001, ssl_tau:0.5, ssl_weight:0.05
  Valid result:
  recall@10 : 0.0953    mrr@10 : 0.072    ndcg@10 : 0.0619    hit@10 : 0.1714    precision@10 : 0.0203
  Test result:
  recall@10 : 0.0984    mrr@10 : 0.0744    ndcg@10 : 0.0638    hit@10 : 0.1769    precision@10 : 0.0212
  ```
  
- **Logging Result**:

  ```yaml
  100%|██████████| 24/24 [131:13:36<00:00, 19684.00s/trial, best loss: -0.0674]
  best params:  {'drop_ratio': 0.2, 'learning_rate': 0.001, 'ssl_tau': 0.2, 'ssl_weight': 0.1}
  best result: 
  {'model': 'SGL', 'best_valid_score': 0.0674, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1037), ('mrr@10', 0.0791), ('ndcg@10', 0.0674), ('hit@10', 0.186), ('precision@10', 0.0223)]), 'test_result': OrderedDict([('recall@10', 0.104), ('mrr@10', 0.0806), ('ndcg@10', 0.0684), ('hit@10', 0.1866), ('precision@10', 0.0226)])}
  ```
