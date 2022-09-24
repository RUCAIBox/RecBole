# General Recommendation

- **Dataset**: [Yelp2022](../../md/yelp_general.md)

- **Model**: [LightGCN](https://recbole.io/docs/user_guide/model/general/lightgcn.html)

- **Time cost**: 48865.55s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [1e-4,5e-4,1e-3,5e-3,0.01]
  n_layers choice [1,2,3,4]
  reg_weight choice [1e-05,1e-04,1e-03]
  ```
  
- **Best parameters**:

  ```yaml
  learning_rate: 1e-3
  n_layers: 2
  reg_weight: 1e-03
  ```
  
- **Hyper-parameter logging**:

  ```yaml
  learning_rate:0.01, n_layers:2, reg_weight:1e-05
  Valid result:
  recall@10 : 0.0712    mrr@10 : 0.0524    ndcg@10 : 0.0452    hit@10 : 0.1306    precision@10 : 0.0148
  Test result:
  recall@10 : 0.0704    mrr@10 : 0.0521    ndcg@10 : 0.0448    hit@10 : 0.1312    precision@10 : 0.015
  
  learning_rate:0.0001, n_layers:3, reg_weight:0.0001
  Valid result:
  recall@10 : 0.0523    mrr@10 : 0.0372    ndcg@10 : 0.032    hit@10 : 0.1016    precision@10 : 0.0113
  Test result:
  recall@10 : 0.0534    mrr@10 : 0.0378    ndcg@10 : 0.0323    hit@10 : 0.1025    precision@10 : 0.0114
  
  learning_rate:0.0005, n_layers:3, reg_weight:0.0001
  Valid result:
  recall@10 : 0.0797    mrr@10 : 0.057    ndcg@10 : 0.0491    hit@10 : 0.1492    precision@10 : 0.0173
  Test result:
  recall@10 : 0.0815    mrr@10 : 0.0588    ndcg@10 : 0.0505    hit@10 : 0.1513    precision@10 : 0.017
  
  learning_rate:0.001, n_layers:3, reg_weight:0.001
  Valid result:
  recall@10 : 0.0906    mrr@10 : 0.0655    ndcg@10 : 0.057    hit@10 : 0.165    precision@10 : 0.0195
  Test result:
  recall@10 : 0.0943    mrr@10 : 0.0692    ndcg@10 : 0.06    hit@10 : 0.1694    precision@10 : 0.0202
  
  learning_rate:0.01, n_layers:3, reg_weight:0.001
  Valid result:
  recall@10 : 0.0758    mrr@10 : 0.0564    ndcg@10 : 0.0487    hit@10 : 0.1395    precision@10 : 0.016
  Test result:
  recall@10 : 0.0771    mrr@10 : 0.0578    ndcg@10 : 0.0494    hit@10 : 0.1419    precision@10 : 0.0164
  
  learning_rate:0.005, n_layers:2, reg_weight:0.0001
  Valid result:
  recall@10 : 0.0815    mrr@10 : 0.0605    ndcg@10 : 0.0525    hit@10 : 0.1479    precision@10 : 0.0171
  Test result:
  recall@10 : 0.0817    mrr@10 : 0.0598    ndcg@10 : 0.052    hit@10 : 0.1485    precision@10 : 0.0172
  
  learning_rate:0.0001, n_layers:3, reg_weight:1e-05
  Valid result:
  recall@10 : 0.0523    mrr@10 : 0.0372    ndcg@10 : 0.032    hit@10 : 0.1016    precision@10 : 0.0113
  Test result:
  recall@10 : 0.0534    mrr@10 : 0.0378    ndcg@10 : 0.0323    hit@10 : 0.1025    precision@10 : 0.0114
  
  learning_rate:0.01, n_layers:2, reg_weight:0.001
  Valid result:
  recall@10 : 0.0715    mrr@10 : 0.0526    ndcg@10 : 0.0454    hit@10 : 0.1311    precision@10 : 0.0149
  Test result:
  recall@10 : 0.0704    mrr@10 : 0.0521    ndcg@10 : 0.0448    hit@10 : 0.1314    precision@10 : 0.015
  
  learning_rate:0.01, n_layers:4, reg_weight:0.0001
  Valid result:
  recall@10 : 0.08    mrr@10 : 0.0601    ndcg@10 : 0.0518    hit@10 : 0.1462    precision@10 : 0.0169
  Test result:
  recall@10 : 0.079    mrr@10 : 0.0597    ndcg@10 : 0.0511    hit@10 : 0.1458    precision@10 : 0.0168
  
  learning_rate:0.0001, n_layers:2, reg_weight:1e-05
  Valid result:
  recall@10 : 0.0557    mrr@10 : 0.0394    ndcg@10 : 0.034    hit@10 : 0.1076    precision@10 : 0.012
  Test result:
  recall@10 : 0.0567    mrr@10 : 0.04    ndcg@10 : 0.0344    hit@10 : 0.1078    precision@10 : 0.0121
  
  learning_rate:0.001, n_layers:2, reg_weight:0.001
  Valid result:
  recall@10 : 0.0921    mrr@10 : 0.0679    ndcg@10 : 0.0589    hit@10 : 0.1668    precision@10 : 0.0197
  Test result:
  recall@10 : 0.0944    mrr@10 : 0.0692    ndcg@10 : 0.0602    hit@10 : 0.1691    precision@10 : 0.0202
  
  learning_rate:0.005, n_layers:1, reg_weight:0.001
  Valid result:
  recall@10 : 0.072    mrr@10 : 0.0528    ndcg@10 : 0.0456    hit@10 : 0.133    precision@10 : 0.0152
  Test result:
  recall@10 : 0.0733    mrr@10 : 0.0539    ndcg@10 : 0.0465    hit@10 : 0.1353    precision@10 : 0.0155
  
  learning_rate:0.001, n_layers:1, reg_weight:0.0001
  Valid result:
  recall@10 : 0.089    mrr@10 : 0.066    ndcg@10 : 0.0571    hit@10 : 0.1606    precision@10 : 0.0188
  Test result:
  recall@10 : 0.0906    mrr@10 : 0.0648    ndcg@10 : 0.057    hit@10 : 0.1626    precision@10 : 0.0193
  
  learning_rate:0.001, n_layers:3, reg_weight:0.0001
  Valid result:
  recall@10 : 0.0902    mrr@10 : 0.0656    ndcg@10 : 0.057    hit@10 : 0.1645    precision@10 : 0.0194
  Test result:
  recall@10 : 0.0939    mrr@10 : 0.069    ndcg@10 : 0.0598    hit@10 : 0.1688    precision@10 : 0.0201
  ```
  
- **Logging Result**:

  ```yaml
  100%|██████████| 60/60 [814:25:33<00:00, 48865.55s/trial, best loss: -0.0589]
  best params:  {'learning_rate': 0.001, 'n_layers': 2, 'reg_weight': 0.001}
  best result: 
  {'model': 'LightGCN', 'best_valid_score': 0.0589, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0921), ('mrr@10', 0.0679), ('ndcg@10', 0.0589), ('hit@10', 0.1668), ('precision@10', 0.0197)]), 'test_result': OrderedDict([('recall@10', 0.0944), ('mrr@10', 0.0692), ('ndcg@10', 0.0602), ('hit@10', 0.1691), ('precision@10', 0.0202)])}
  ```
