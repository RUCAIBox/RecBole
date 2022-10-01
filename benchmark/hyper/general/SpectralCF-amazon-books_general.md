# General Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_general.md)

- **Model**: [SLIMElastic](https://recbole.io/docs/user_guide/model/general/slimelastic.html)

- **Time cost**: 8692.04s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate in [0.002,0.001,0.0005] 
  reg_weight in [0.002,0.001,0.0005] 
  n_layers in [1,2,3,4]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.002
  reg_weight: 0.0005
  n_layers: 2
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.002, n_layers:3, reg_weight:0.0005
  Valid result:
  recall@10 : 0.1084    mrr@10 : 0.0742    ndcg@10 : 0.0674    hit@10 : 0.1837    precision@10 : 0.0216
  Test result:
  recall@10 : 0.1096    mrr@10 : 0.0797    ndcg@10 : 0.0712    hit@10 : 0.1866    precision@10 : 0.0226

  learning_rate:0.001, n_layers:4, reg_weight:0.001
  Valid result:
  recall@10 : 0.1075    mrr@10 : 0.0738    ndcg@10 : 0.0668    hit@10 : 0.1847    precision@10 : 0.0219
  Test result:
  recall@10 : 0.1084    mrr@10 : 0.0776    ndcg@10 : 0.0697    hit@10 : 0.1857    precision@10 : 0.0227

  learning_rate:0.0005, n_layers:3, reg_weight:0.0005
  Valid result:
  recall@10 : 0.0112    mrr@10 : 0.0086    ndcg@10 : 0.0081    hit@10 : 0.0196    precision@10 : 0.002
  Test result:
  recall@10 : 0.01    mrr@10 : 0.0077    ndcg@10 : 0.0071    hit@10 : 0.0181    precision@10 : 0.0018

  learning_rate:0.001, n_layers:2, reg_weight:0.001
  Valid result:
  recall@10 : 0.1108    mrr@10 : 0.0759    ndcg@10 : 0.069    hit@10 : 0.1885    precision@10 : 0.0226
  Test result:
  recall@10 : 0.112    mrr@10 : 0.0803    ndcg@10 : 0.0726    hit@10 : 0.1891    precision@10 : 0.0232

  learning_rate:0.0005, n_layers:2, reg_weight:0.001
  Valid result:
  recall@10 : 0.0118    mrr@10 : 0.0086    ndcg@10 : 0.0082    hit@10 : 0.0195    precision@10 : 0.002
  Test result:
  recall@10 : 0.0104    mrr@10 : 0.0077    ndcg@10 : 0.0071    hit@10 : 0.0178    precision@10 : 0.0018

  learning_rate:0.0005, n_layers:2, reg_weight:0.002
  Valid result:
  recall@10 : 0.0117    mrr@10 : 0.0086    ndcg@10 : 0.0082    hit@10 : 0.0194    precision@10 : 0.002
  Test result:
  recall@10 : 0.0104    mrr@10 : 0.0077    ndcg@10 : 0.0071    hit@10 : 0.0177    precision@10 : 0.0018

  learning_rate:0.0005, n_layers:1, reg_weight:0.0005
  Valid result:
  recall@10 : 0.0528    mrr@10 : 0.0387    ndcg@10 : 0.0328    hit@10 : 0.1042    precision@10 : 0.0118
  Test result:
  recall@10 : 0.0538    mrr@10 : 0.0394    ndcg@10 : 0.0334    hit@10 : 0.1056    precision@10 : 0.0122

  learning_rate:0.001, n_layers:3, reg_weight:0.001
  Valid result:
  recall@10 : 0.1061    mrr@10 : 0.0739    ndcg@10 : 0.0665    hit@10 : 0.1817    precision@10 : 0.0216
  Test result:
  recall@10 : 0.1076    mrr@10 : 0.0779    ndcg@10 : 0.0693    hit@10 : 0.1842    precision@10 : 0.0225

  learning_rate:0.0005, n_layers:4, reg_weight:0.002
  Valid result:
  recall@10 : 0.0116    mrr@10 : 0.0086    ndcg@10 : 0.0082    hit@10 : 0.0196    precision@10 : 0.002
  Test result:
  recall@10 : 0.0101    mrr@10 : 0.0077    ndcg@10 : 0.0071    hit@10 : 0.018    precision@10 : 0.0018

  learning_rate:0.001, n_layers:2, reg_weight:0.002
  Valid result:
  recall@10 : 0.1017    mrr@10 : 0.0698    ndcg@10 : 0.0633    hit@10 : 0.1757    precision@10 : 0.0209
  Test result:
  recall@10 : 0.1038    mrr@10 : 0.0734    ndcg@10 : 0.066    hit@10 : 0.1781    precision@10 : 0.0217

  learning_rate:0.001, n_layers:1, reg_weight:0.002
  Valid result:
  recall@10 : 0.0824    mrr@10 : 0.0572    ndcg@10 : 0.051    hit@10 : 0.1462    precision@10 : 0.0171
  Test result:
  recall@10 : 0.0834    mrr@10 : 0.0601    ndcg@10 : 0.0524    hit@10 : 0.149    precision@10 : 0.0179

  learning_rate:0.002, n_layers:1, reg_weight:0.0005
  Valid result:
  recall@10 : 0.1143    mrr@10 : 0.0789    ndcg@10 : 0.0719    hit@10 : 0.1918    precision@10 : 0.0226
  Test result:
  recall@10 : 0.1171    mrr@10 : 0.0838    ndcg@10 : 0.0754    hit@10 : 0.1957    precision@10 : 0.0236

  learning_rate:0.001, n_layers:4, reg_weight:0.0005
  Valid result:
  recall@10 : 0.1102    mrr@10 : 0.0757    ndcg@10 : 0.0688    hit@10 : 0.188    precision@10 : 0.0224
  Test result:
  recall@10 : 0.1113    mrr@10 : 0.0797    ndcg@10 : 0.0716    hit@10 : 0.1889    precision@10 : 0.0232

  learning_rate:0.001, n_layers:1, reg_weight:0.0005
  Valid result:
  recall@10 : 0.0828    mrr@10 : 0.057    ndcg@10 : 0.051    hit@10 : 0.1473    precision@10 : 0.0172
  Test result:
  recall@10 : 0.0827    mrr@10 : 0.0596    ndcg@10 : 0.052    hit@10 : 0.148    precision@10 : 0.0177

  learning_rate:0.002, n_layers:3, reg_weight:0.002
  Valid result:
  recall@10 : 0.1087    mrr@10 : 0.0754    ndcg@10 : 0.0683    hit@10 : 0.1834    precision@10 : 0.0216
  Test result:
  recall@10 : 0.1113    mrr@10 : 0.0798    ndcg@10 : 0.0717    hit@10 : 0.1871    precision@10 : 0.0227

  learning_rate:0.001, n_layers:2, reg_weight:0.0005
  Valid result:
  recall@10 : 0.1055    mrr@10 : 0.0717    ndcg@10 : 0.0655    hit@10 : 0.1795    precision@10 : 0.0214
  Test result:
  recall@10 : 0.1053    mrr@10 : 0.075    ndcg@10 : 0.0675    hit@10 : 0.1796    precision@10 : 0.0221

  learning_rate:0.002, n_layers:1, reg_weight:0.002
  Valid result:
  recall@10 : 0.1153    mrr@10 : 0.08    ndcg@10 : 0.0728    hit@10 : 0.1927    precision@10 : 0.0227
  Test result:
  recall@10 : 0.1175    mrr@10 : 0.0845    ndcg@10 : 0.0761    hit@10 : 0.1967    precision@10 : 0.0237

  learning_rate:0.0005, n_layers:4, reg_weight:0.001
  Valid result:
  recall@10 : 0.0116    mrr@10 : 0.0086    ndcg@10 : 0.0082    hit@10 : 0.0197    precision@10 : 0.002
  Test result:
  recall@10 : 0.0104    mrr@10 : 0.0078    ndcg@10 : 0.0072    hit@10 : 0.0182    precision@10 : 0.0019

  learning_rate:0.002, n_layers:2, reg_weight:0.0005
  Valid result:
  recall@10 : 0.1196    mrr@10 : 0.082    ndcg@10 : 0.0753    hit@10 : 0.1999    precision@10 : 0.0236
  Test result:
  recall@10 : 0.1219    mrr@10 : 0.0871    ndcg@10 : 0.079    hit@10 : 0.2024    precision@10 : 0.0244

  learning_rate:0.002, n_layers:4, reg_weight:0.002
  Valid result:
  recall@10 : 0.1167    mrr@10 : 0.0805    ndcg@10 : 0.0735    hit@10 : 0.1946    precision@10 : 0.0232
  Test result:
  recall@10 : 0.1189    mrr@10 : 0.0854    ndcg@10 : 0.0774    hit@10 : 0.1987    precision@10 : 0.0239

  learning_rate:0.002, n_layers:1, reg_weight:0.001
  Valid result:
  recall@10 : 0.1148    mrr@10 : 0.0787    ndcg@10 : 0.072    hit@10 : 0.1923    precision@10 : 0.0226
  Test result:
  recall@10 : 0.1173    mrr@10 : 0.084    ndcg@10 : 0.0756    hit@10 : 0.1961    precision@10 : 0.0236

  learning_rate:0.0005, n_layers:4, reg_weight:0.0005
  Valid result:
  recall@10 : 0.0117    mrr@10 : 0.0087    ndcg@10 : 0.0082    hit@10 : 0.0198    precision@10 : 0.002
  Test result:
  recall@10 : 0.0104    mrr@10 : 0.0078    ndcg@10 : 0.0072    hit@10 : 0.0182    precision@10 : 0.0019

  learning_rate:0.001, n_layers:3, reg_weight:0.0005
  Valid result:
  recall@10 : 0.081    mrr@10 : 0.0558    ndcg@10 : 0.0498    hit@10 : 0.1451    precision@10 : 0.0171
  Test result:
  recall@10 : 0.0792    mrr@10 : 0.0578    ndcg@10 : 0.0505    hit@10 : 0.1427    precision@10 : 0.0173

  learning_rate:0.0005, n_layers:3, reg_weight:0.002
  Valid result:
  recall@10 : 0.0112    mrr@10 : 0.0086    ndcg@10 : 0.0081    hit@10 : 0.0196    precision@10 : 0.002
  Test result:
  recall@10 : 0.01    mrr@10 : 0.0077    ndcg@10 : 0.0071    hit@10 : 0.018    precision@10 : 0.0018

  learning_rate:0.0005, n_layers:1, reg_weight:0.002
  Valid result:
  recall@10 : 0.0611    mrr@10 : 0.0432    ndcg@10 : 0.0376    hit@10 : 0.1162    precision@10 : 0.0134
  Test result:
  recall@10 : 0.0605    mrr@10 : 0.0425    ndcg@10 : 0.037    hit@10 : 0.1147    precision@10 : 0.0136

  learning_rate:0.001, n_layers:1, reg_weight:0.001
  Valid result:
  recall@10 : 0.0825    mrr@10 : 0.057    ndcg@10 : 0.051    hit@10 : 0.1468    precision@10 : 0.0171
  Test result:
  recall@10 : 0.0831    mrr@10 : 0.0599    ndcg@10 : 0.0522    hit@10 : 0.1488    precision@10 : 0.0178

  learning_rate:0.001, n_layers:3, reg_weight:0.002
  Valid result:
  recall@10 : 0.0999    mrr@10 : 0.0693    ndcg@10 : 0.0626    hit@10 : 0.1728    precision@10 : 0.0206
  Test result:
  recall@10 : 0.1006    mrr@10 : 0.0714    ndcg@10 : 0.0641    hit@10 : 0.1734    precision@10 : 0.0213

  learning_rate:0.002, n_layers:2, reg_weight:0.001
  Valid result:
  recall@10 : 0.1149    mrr@10 : 0.0786    ndcg@10 : 0.0721    hit@10 : 0.1938    precision@10 : 0.0228
  Test result:
  recall@10 : 0.1168    mrr@10 : 0.0827    ndcg@10 : 0.0751    hit@10 : 0.196    precision@10 : 0.0236

  learning_rate:0.0005, n_layers:1, reg_weight:0.001
  Valid result:
  recall@10 : 0.0528    mrr@10 : 0.0387    ndcg@10 : 0.0328    hit@10 : 0.1042    precision@10 : 0.0119
  Test result:
  recall@10 : 0.0537    mrr@10 : 0.0392    ndcg@10 : 0.0333    hit@10 : 0.1053    precision@10 : 0.0122
  ```

- **Logging Result**:

  ```yaml
  INFO  Early stop triggered. Stopping iterations as condition is reach.
  81%|████████  | 29/36 [70:01:09<16:54:04, 8692.04s/trial, best loss: -0.0753]
  best params:  {'learning_rate': 0.002, 'n_layers': 2, 'reg_weight': 0.0005}
  best result: 
  {'model': 'SpectralCF', 'best_valid_score': 0.0753, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1196), ('mrr@10', 0.082), ('ndcg@10', 0.0753), ('hit@10', 0.1999), ('precision@10', 0.0236)]), 'test_result': OrderedDict([('recall@10', 0.1219), ('mrr@10', 0.0871), ('ndcg@10', 0.079), ('hit@10', 0.2024), ('precision@10', 0.0244)])}
  ```
