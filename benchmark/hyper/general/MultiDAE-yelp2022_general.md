# General Recommendation

- **Dataset**: [Yelp2022](../../md/yelp_general.md)

- **Model**: [MultiDAE](https://recbole.io/docs/user_guide/model/general/multidae.html)

- **Time cost**: 3841.35s/trial

- **Hyper-parameter searching** (hyper.test):s

  ```yaml
  learning_rate choice [1e-4,5e-4,1e-3,0.0015,5e-3,0.01]
  ```
  
- **Best parameters**:

  ```yaml
  learning_rate: 0.0015
  ```
  
- **Hyper-parameter logging**:

  ```yaml
  learning_rate:0.0015
  Valid result:
  recall@10 : 0.1086    mrr@10 : 0.14    ndcg@10 : 0.0978    hit@10 : 0.2004    precision@10 : 0.0275
  Test result:
  recall@10 : 0.1103    mrr@10 : 0.1408    ndcg@10 : 0.0988    hit@10 : 0.2019    precision@10 : 0.0276
  
  learning_rate:0.0005
  Valid result:
  recall@10 : 0.1114    mrr@10 : 0.137    ndcg@10 : 0.0978    hit@10 : 0.2037    precision@10 : 0.0277
  Test result:
  recall@10 : 0.1106    mrr@10 : 0.1377    ndcg@10 : 0.0978    hit@10 : 0.2021    precision@10 : 0.0276
  
  learning_rate:0.01
  Valid result:
  recall@10 : 0.1021    mrr@10 : 0.0991    ndcg@10 : 0.0779    hit@10 : 0.1828    precision@10 : 0.0223
  Test result:
  recall@10 : 0.1019    mrr@10 : 0.0992    ndcg@10 : 0.0781    hit@10 : 0.1835    precision@10 : 0.0225
  
  learning_rate:0.0001
  Valid result:
  recall@10 : 0.0996    mrr@10 : 0.0823    ndcg@10 : 0.0676    hit@10 : 0.1811    precision@10 : 0.0219
  Test result:
  recall@10 : 0.0999    mrr@10 : 0.084    ndcg@10 : 0.0686    hit@10 : 0.1809    precision@10 : 0.0222
  
  learning_rate:0.001
  Valid result:
  recall@10 : 0.1093    mrr@10 : 0.1368    ndcg@10 : 0.0969    hit@10 : 0.2015    precision@10 : 0.0273
  Test result:
  recall@10 : 0.1108    mrr@10 : 0.1375    ndcg@10 : 0.0977    hit@10 : 0.2021    precision@10 : 0.0275
  
  learning_rate:0.005
  Valid result:
  recall@10 : 0.1069    mrr@10 : 0.1213    ndcg@10 : 0.0888    hit@10 : 0.1944    precision@10 : 0.0253
  Test result:
  recall@10 : 0.107    mrr@10 : 0.1207    ndcg@10 : 0.0888    hit@10 : 0.1948    precision@10 : 0.0254
  ```
  
- **Logging Result**:

  ```yaml
  100%|██████████| 6/6 [6:24:08<00:00, 3841.35s/trial, best loss: -0.0978]
  best params:  {'learning_rate': 0.0015}
  best result: 
  {'model': 'MultiDAE', 'best_valid_score': 0.0978, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1086), ('mrr@10', 0.14), ('ndcg@10', 0.0978), ('hit@10', 0.2004), ('precision@10', 0.0275)]), 'test_result': OrderedDict([('recall@10', 0.1103), ('mrr@10', 0.1408), ('ndcg@10', 0.0988), ('hit@10', 0.2019), ('precision@10', 0.0276)])}
  ```
