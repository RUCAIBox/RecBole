# General Recommendation

- **Dataset**: [Yelp2022](../../md/yelp_general.md)

- **Model**: [MultiVAE](https://recbole.io/docs/user_guide/model/general/multivae.html)

- **Time cost**: 3953.86s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [1e-4,5e-4,1e-3,0.0015,3e-3,5e-3,0.01]
  ```
  
- **Best parameters**:

  ```yaml
  learning_rate: 5e-4
  ```
  
- **Hyper-parameter logging**:

  ```yaml
  learning_rate:0.0001
  Valid result:
  recall@10 : 0.0967    mrr@10 : 0.0776    ndcg@10 : 0.0641    hit@10 : 0.1776    precision@10 : 0.0217
  Test result:
  recall@10 : 0.0974    mrr@10 : 0.0787    ndcg@10 : 0.065    hit@10 : 0.1784    precision@10 : 0.0221
  
  learning_rate:0.003
  Valid result:
  recall@10 : 0.11    mrr@10 : 0.135    ndcg@10 : 0.0962    hit@10 : 0.2008    precision@10 : 0.0269
  Test result:
  recall@10 : 0.1101    mrr@10 : 0.1342    ndcg@10 : 0.096    hit@10 : 0.2002    precision@10 : 0.0268
  
  learning_rate:0.0005
  Valid result:
  recall@10 : 0.1143    mrr@10 : 0.1401    ndcg@10 : 0.1002    hit@10 : 0.2078    precision@10 : 0.0281
  Test result:
  recall@10 : 0.1151    mrr@10 : 0.1412    ndcg@10 : 0.1012    hit@10 : 0.2085    precision@10 : 0.0283
  
  learning_rate:0.01
  Valid result:
  recall@10 : 0.1039    mrr@10 : 0.1029    ndcg@10 : 0.0796    hit@10 : 0.1875    precision@10 : 0.0231
  Test result:
  recall@10 : 0.1042    mrr@10 : 0.1024    ndcg@10 : 0.0798    hit@10 : 0.1871    precision@10 : 0.0231
  
  learning_rate:0.001
  Valid result:
  recall@10 : 0.1127    mrr@10 : 0.1363    ndcg@10 : 0.0981    hit@10 : 0.2052    precision@10 : 0.0276
  Test result:
  recall@10 : 0.1136    mrr@10 : 0.1365    ndcg@10 : 0.0984    hit@10 : 0.2061    precision@10 : 0.0277
  
  learning_rate:0.0015
  Valid result:
  recall@10 : 0.1127    mrr@10 : 0.1371    ndcg@10 : 0.0983    hit@10 : 0.2047    precision@10 : 0.0276
  Test result:
  recall@10 : 0.1136    mrr@10 : 0.1381    ndcg@10 : 0.099    hit@10 : 0.2062    precision@10 : 0.0278
  
  learning_rate:0.005
  Valid result:
  recall@10 : 0.1081    mrr@10 : 0.1255    ndcg@10 : 0.0913    hit@10 : 0.1972    precision@10 : 0.0259
  Test result:
  recall@10 : 0.1093    mrr@10 : 0.1265    ndcg@10 : 0.0922    hit@10 : 0.1991    precision@10 : 0.0261
  ```
  
- **Logging Result**:

  ```yaml
  100%|██████████| 7/7 [7:41:17<00:00, 3953.86s/trial, best loss: -0.1002]
  best params:  {'learning_rate': 0.0005}
  best result: 
  {'model': 'MultiVAE', 'best_valid_score': 0.1002, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1143), ('mrr@10', 0.1401), ('ndcg@10', 0.1002), ('hit@10', 0.2078), ('precision@10', 0.0281)]), 'test_result': OrderedDict([('recall@10', 0.1151), ('mrr@10', 0.1412), ('ndcg@10', 0.1012), ('hit@10', 0.2085), ('precision@10', 0.0283)])}
  ```
