# General Recommendation

- **Dataset**: [Yelp2022](../../md/yelp_general.md)

- **Model**: [ENMF](https://recbole.io/docs/user_guide/model/general/enmf.html)

- **Time cost**: 4916.30s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [1e-4,1e-3,5e-3]
  dropout_prob choice [0.5,0.3,0.1]
  negative_weight choice [0.001,0.005,0.01,0.05]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  dropout_prob: 0.3
  negative_weight: 0.05
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  dropout_prob:0.1, learning_rate:0.001, negative_weight:0.001
  Valid result:
  recall@10 : 0.0575    mrr@10 : 0.0313    ndcg@10 : 0.0322    hit@10 : 0.0911    precision@10 : 0.0095
  Test result:
  recall@10 : 0.0575    mrr@10 : 0.0307    ndcg@10 : 0.0317    hit@10 : 0.092    precision@10 : 0.0096
  
  dropout_prob:0.3, learning_rate:0.001, negative_weight:0.05
  Valid result:
  recall@10 : 0.0786    mrr@10 : 0.0569    ndcg@10 : 0.0488    hit@10 : 0.1471    precision@10 : 0.0169
  Test result:
  recall@10 : 0.0795    mrr@10 : 0.0592    ndcg@10 : 0.0501    hit@10 : 0.1488    precision@10 : 0.0171
  
  dropout_prob:0.5, learning_rate:0.001, negative_weight:0.01
  Valid result:
  recall@10 : 0.0767    mrr@10 : 0.0518    ndcg@10 : 0.0466    hit@10 : 0.1385    precision@10 : 0.0154
  Test result:
  recall@10 : 0.0779    mrr@10 : 0.0531    ndcg@10 : 0.0473    hit@10 : 0.1411    precision@10 : 0.0157
  
  dropout_prob:0.5, learning_rate:0.0001, negative_weight:0.01
  Valid result:
  recall@10 : 0.0371    mrr@10 : 0.0293    ndcg@10 : 0.0237    hit@10 : 0.0762    precision@10 : 0.0084
  Test result:
  recall@10 : 0.0381    mrr@10 : 0.0294    ndcg@10 : 0.0239    hit@10 : 0.076    precision@10 : 0.0084
  
  dropout_prob:0.3, learning_rate:0.0001, negative_weight:0.001
  Valid result:
  recall@10 : 0.0386    mrr@10 : 0.0302    ndcg@10 : 0.0247    hit@10 : 0.0781    precision@10 : 0.0086
  Test result:
  recall@10 : 0.0392    mrr@10 : 0.03    ndcg@10 : 0.0246    hit@10 : 0.0776    precision@10 : 0.0085
  
  dropout_prob:0.5, learning_rate:0.0001, negative_weight:0.05
  Valid result:
  recall@10 : 0.0709    mrr@10 : 0.0496    ndcg@10 : 0.0436    hit@10 : 0.1293    precision@10 : 0.0144
  Test result:
  recall@10 : 0.0727    mrr@10 : 0.0511    ndcg@10 : 0.0448    hit@10 : 0.1312    precision@10 : 0.0147
  
  dropout_prob:0.1, learning_rate:0.0001, negative_weight:0.001
  Valid result:
  recall@10 : 0.0378    mrr@10 : 0.0295    ndcg@10 : 0.0241    hit@10 : 0.0762    precision@10 : 0.0084
  Test result:
  recall@10 : 0.0385    mrr@10 : 0.0299    ndcg@10 : 0.0243    hit@10 : 0.0765    precision@10 : 0.0084
  
  dropout_prob:0.5, learning_rate:0.001, negative_weight:0.005
  Valid result:
  recall@10 : 0.0756    mrr@10 : 0.048    ndcg@10 : 0.0448    hit@10 : 0.1326    precision@10 : 0.0145
  Test result:
  recall@10 : 0.0758    mrr@10 : 0.0493    ndcg@10 : 0.0453    hit@10 : 0.1343    precision@10 : 0.0147
  
  dropout_prob:0.1, learning_rate:0.005, negative_weight:0.05
  Valid result:
  recall@10 : 0.0773    mrr@10 : 0.0567    ndcg@10 : 0.0483    hit@10 : 0.1465    precision@10 : 0.0169
  Test result:
  recall@10 : 0.0787    mrr@10 : 0.0586    ndcg@10 : 0.0495    hit@10 : 0.1484    precision@10 : 0.0171
  
  dropout_prob:0.5, learning_rate:0.001, negative_weight:0.05
  Valid result:
  recall@10 : 0.0763    mrr@10 : 0.0554    ndcg@10 : 0.0476    hit@10 : 0.1427    precision@10 : 0.0163
  Test result:
  recall@10 : 0.0788    mrr@10 : 0.0585    ndcg@10 : 0.0497    hit@10 : 0.1464    precision@10 : 0.0168
  
  dropout_prob:0.5, learning_rate:0.0001, negative_weight:0.005
  Valid result:
  recall@10 : 0.04    mrr@10 : 0.0294    ndcg@10 : 0.025    hit@10 : 0.0779    precision@10 : 0.0085
  Test result:
  recall@10 : 0.0416    mrr@10 : 0.0295    ndcg@10 : 0.0254    hit@10 : 0.079    precision@10 : 0.0086
  
  dropout_prob:0.1, learning_rate:0.001, negative_weight:0.005
  Valid result:
  recall@10 : 0.0749    mrr@10 : 0.0478    ndcg@10 : 0.0446    hit@10 : 0.1305    precision@10 : 0.0141
  Test result:
  recall@10 : 0.0753    mrr@10 : 0.0486    ndcg@10 : 0.0449    hit@10 : 0.132    precision@10 : 0.0144
  
  dropout_prob:0.1, learning_rate:0.005, negative_weight:0.01
  Valid result:
  recall@10 : 0.0796    mrr@10 : 0.053    ndcg@10 : 0.048    hit@10 : 0.1439    precision@10 : 0.016
  Test result:
  recall@10 : 0.0811    mrr@10 : 0.0544    ndcg@10 : 0.0492    hit@10 : 0.1454    precision@10 : 0.0162
  
  dropout_prob:0.5, learning_rate:0.005, negative_weight:0.05
  Valid result:
  recall@10 : 0.076    mrr@10 : 0.0559    ndcg@10 : 0.0475    hit@10 : 0.1436    precision@10 : 0.0165
  Test result:
  recall@10 : 0.0778    mrr@10 : 0.0584    ndcg@10 : 0.0492    hit@10 : 0.1459    precision@10 : 0.0167
  
  dropout_prob:0.3, learning_rate:0.005, negative_weight:0.01
  Valid result:
  recall@10 : 0.0785    mrr@10 : 0.0529    ndcg@10 : 0.0475    hit@10 : 0.1417    precision@10 : 0.0158
  Test result:
  recall@10 : 0.0804    mrr@10 : 0.0545    ndcg@10 : 0.0488    hit@10 : 0.1443    precision@10 : 0.016
  
  dropout_prob:0.5, learning_rate:0.005, negative_weight:0.01
  Valid result:
  recall@10 : 0.0777    mrr@10 : 0.0527    ndcg@10 : 0.0472    hit@10 : 0.1415    precision@10 : 0.0157
  Test result:
  recall@10 : 0.0799    mrr@10 : 0.055    ndcg@10 : 0.0488    hit@10 : 0.1443    precision@10 : 0.0161
  
  dropout_prob:0.1, learning_rate:0.0001, negative_weight:0.01
  Valid result:
  recall@10 : 0.0363    mrr@10 : 0.0288    ndcg@10 : 0.0232    hit@10 : 0.0758    precision@10 : 0.0084
  Test result:
  recall@10 : 0.0375    mrr@10 : 0.0292    ndcg@10 : 0.0235    hit@10 : 0.0764    precision@10 : 0.0085
  
  dropout_prob:0.5, learning_rate:0.001, negative_weight:0.001
  Valid result:
  recall@10 : 0.0403    mrr@10 : 0.0307    ndcg@10 : 0.0254    hit@10 : 0.0803    precision@10 : 0.0089
  Test result:
  recall@10 : 0.0415    mrr@10 : 0.0306    ndcg@10 : 0.0255    hit@10 : 0.0818    precision@10 : 0.009
  
  dropout_prob:0.1, learning_rate:0.005, negative_weight:0.001
  Valid result:
  recall@10 : 0.0482    mrr@10 : 0.0264    ndcg@10 : 0.0269    hit@10 : 0.0788    precision@10 : 0.0082
  Test result:
  recall@10 : 0.048    mrr@10 : 0.0265    ndcg@10 : 0.0268    hit@10 : 0.079    precision@10 : 0.0082
  
  dropout_prob:0.1, learning_rate:0.0001, negative_weight:0.005
  Valid result:
  recall@10 : 0.067    mrr@10 : 0.045    ndcg@10 : 0.0408    hit@10 : 0.1197    precision@10 : 0.0131
  Test result:
  recall@10 : 0.0683    mrr@10 : 0.0456    ndcg@10 : 0.0412    hit@10 : 0.1212    precision@10 : 0.0132
  
  dropout_prob:0.3, learning_rate:0.001, negative_weight:0.01
  Valid result:
  recall@10 : 0.0801    mrr@10 : 0.0534    ndcg@10 : 0.0482    hit@10 : 0.1432    precision@10 : 0.0158
  Test result:
  recall@10 : 0.0806    mrr@10 : 0.0547    ndcg@10 : 0.049    hit@10 : 0.1446    precision@10 : 0.0161
  
  dropout_prob:0.5, learning_rate:0.0001, negative_weight:0.001
  Valid result:
  recall@10 : 0.038    mrr@10 : 0.0291    ndcg@10 : 0.024    hit@10 : 0.0757    precision@10 : 0.0083
  Test result:
  recall@10 : 0.0391    mrr@10 : 0.0294    ndcg@10 : 0.0244    hit@10 : 0.0763    precision@10 : 0.0084
  
  dropout_prob:0.3, learning_rate:0.0001, negative_weight:0.01
  Valid result:
  recall@10 : 0.0654    mrr@10 : 0.0445    ndcg@10 : 0.0398    hit@10 : 0.1178    precision@10 : 0.0129
  Test result:
  recall@10 : 0.0665    mrr@10 : 0.0457    ndcg@10 : 0.0406    hit@10 : 0.119    precision@10 : 0.0131
  
  dropout_prob:0.1, learning_rate:0.0001, negative_weight:0.05
  Valid result:
  recall@10 : 0.0771    mrr@10 : 0.0552    ndcg@10 : 0.0479    hit@10 : 0.1428    precision@10 : 0.0162
  Test result:
  recall@10 : 0.0791    mrr@10 : 0.0581    ndcg@10 : 0.0497    hit@10 : 0.1461    precision@10 : 0.0167
  
  dropout_prob:0.1, learning_rate:0.001, negative_weight:0.01
  Valid result:
  recall@10 : 0.0793    mrr@10 : 0.0534    ndcg@10 : 0.0481    hit@10 : 0.1425    precision@10 : 0.0158
  Test result:
  recall@10 : 0.0808    mrr@10 : 0.0544    ndcg@10 : 0.0489    hit@10 : 0.1446    precision@10 : 0.016
  
  dropout_prob:0.3, learning_rate:0.005, negative_weight:0.005
  Valid result:
  recall@10 : 0.075    mrr@10 : 0.0474    ndcg@10 : 0.0442    hit@10 : 0.1308    precision@10 : 0.0142
  Test result:
  recall@10 : 0.0758    mrr@10 : 0.0472    ndcg@10 : 0.0443    hit@10 : 0.1316    precision@10 : 0.0143
  
  dropout_prob:0.5, learning_rate:0.005, negative_weight:0.005
  Valid result:
  recall@10 : 0.074    mrr@10 : 0.0464    ndcg@10 : 0.0435    hit@10 : 0.1297    precision@10 : 0.0141
  Test result:
  recall@10 : 0.0749    mrr@10 : 0.0482    ndcg@10 : 0.0446    hit@10 : 0.1317    precision@10 : 0.0144
  
  dropout_prob:0.1, learning_rate:0.001, negative_weight:0.05
  Valid result:
  recall@10 : 0.0781    mrr@10 : 0.0572    ndcg@10 : 0.0488    hit@10 : 0.1466    precision@10 : 0.0169
  Test result:
  recall@10 : 0.0795    mrr@10 : 0.06    ndcg@10 : 0.0505    hit@10 : 0.1486    precision@10 : 0.0172
  
  dropout_prob:0.3, learning_rate:0.001, negative_weight:0.005
  Valid result:
  recall@10 : 0.0734    mrr@10 : 0.0476    ndcg@10 : 0.0441    hit@10 : 0.1291    precision@10 : 0.0141
  Test result:
  recall@10 : 0.0744    mrr@10 : 0.0481    ndcg@10 : 0.0443    hit@10 : 0.1311    precision@10 : 0.0143
  
  dropout_prob:0.3, learning_rate:0.005, negative_weight:0.001
  Valid result:
  recall@10 : 0.0577    mrr@10 : 0.0311    ndcg@10 : 0.0318    hit@10 : 0.0931    precision@10 : 0.0097
  Test result:
  recall@10 : 0.0562    mrr@10 : 0.03    ndcg@10 : 0.0307    hit@10 : 0.0913    precision@10 : 0.0095
  
  dropout_prob:0.3, learning_rate:0.0001, negative_weight:0.05
  Valid result:
  recall@10 : 0.0742    mrr@10 : 0.0528    ndcg@10 : 0.0459    hit@10 : 0.1364    precision@10 : 0.0154
  Test result:
  recall@10 : 0.0765    mrr@10 : 0.0553    ndcg@10 : 0.0479    hit@10 : 0.1403    precision@10 : 0.0159
  
  dropout_prob:0.3, learning_rate:0.005, negative_weight:0.05
  Valid result:
  recall@10 : 0.0768    mrr@10 : 0.0562    ndcg@10 : 0.048    hit@10 : 0.1453    precision@10 : 0.0167
  Test result:
  recall@10 : 0.0786    mrr@10 : 0.0583    ndcg@10 : 0.0494    hit@10 : 0.1472    precision@10 : 0.0169
  
  dropout_prob:0.3, learning_rate:0.0001, negative_weight:0.005
  Valid result:
  recall@10 : 0.0576    mrr@10 : 0.0388    ndcg@10 : 0.0349    hit@10 : 0.1036    precision@10 : 0.0112
  Test result:
  recall@10 : 0.0585    mrr@10 : 0.0398    ndcg@10 : 0.0356    hit@10 : 0.1048    precision@10 : 0.0114
  
  dropout_prob:0.3, learning_rate:0.001, negative_weight:0.001
  Valid result:
  recall@10 : 0.0406    mrr@10 : 0.0285    ndcg@10 : 0.0248    hit@10 : 0.0783    precision@10 : 0.0085
  Test result:
  recall@10 : 0.0405    mrr@10 : 0.0278    ndcg@10 : 0.0242    hit@10 : 0.0774    precision@10 : 0.0084
  
  dropout_prob:0.5, learning_rate:0.005, negative_weight:0.001
  Valid result:
  recall@10 : 0.0554    mrr@10 : 0.0305    ndcg@10 : 0.0307    hit@10 : 0.092    precision@10 : 0.0096
  Test result:
  recall@10 : 0.0549    mrr@10 : 0.0303    ndcg@10 : 0.0304    hit@10 : 0.0918    precision@10 : 0.0097
  
  dropout_prob:0.1, learning_rate:0.005, negative_weight:0.005
  Valid result:
  recall@10 : 0.0765    mrr@10 : 0.0465    ndcg@10 : 0.0444    hit@10 : 0.1325    precision@10 : 0.0144
  Test result:
  recall@10 : 0.0783    mrr@10 : 0.0488    ndcg@10 : 0.0459    hit@10 : 0.1356    precision@10 : 0.0148
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 36/36 [49:09:46<00:00, 4916.30s/trial, best loss: -0.0488]
  best params:  {'dropout_prob': 0.3, 'learning_rate': 0.001, 'negative_weight': 0.05}
  best result: 
  {'model': 'ENMF', 'best_valid_score': 0.0488, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0786), ('mrr@10', 0.0569), ('ndcg@10', 0.0488), ('hit@10', 0.1471), ('precision@10', 0.0169)]), 'test_result': OrderedDict([('recall@10', 0.0795), ('mrr@10', 0.0592), ('ndcg@10', 0.0501), ('hit@10', 0.1488), ('precision@10', 0.0171)])}
  ```
