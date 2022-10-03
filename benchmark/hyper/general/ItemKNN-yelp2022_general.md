# General Recommendation

- **Dataset**: [Yelp2022](../../md/yelp_general.md)

- **Model**: [ItemKNN](https://recbole.io/docs/user_guide/model/general/itemknn.html)

- **Time cost**: 1059.68s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  k choice [10,50,100,200,250,300,400]
  shrink choice [0.0,0.5,1.0]
  ```
  
- **Best parameters**:

  ```yaml
  k: 400
  shrink: 1.0
  ```
  
- **Hyper-parameter logging**:

  ```yaml
  k:50, shrink:0.0
  Valid result:
  recall@10 : 0.0703    mrr@10 : 0.0696    ndcg@10 : 0.0508    hit@10 : 0.1439    precision@10 : 0.0192
  Test result:
  recall@10 : 0.072    mrr@10 : 0.0693    ndcg@10 : 0.0512    hit@10 : 0.1448    precision@10 : 0.0193
  
  k:300, shrink:0.5
  Valid result:
  recall@10 : 0.0817    mrr@10 : 0.0825    ndcg@10 : 0.0598    hit@10 : 0.1645    precision@10 : 0.0223
  Test result:
  recall@10 : 0.0835    mrr@10 : 0.0837    ndcg@10 : 0.0611    hit@10 : 0.1654    precision@10 : 0.0224
  
  k:300, shrink:0.0
  Valid result:
  recall@10 : 0.0812    mrr@10 : 0.0825    ndcg@10 : 0.0596    hit@10 : 0.1638    precision@10 : 0.0222
  Test result:
  recall@10 : 0.0829    mrr@10 : 0.0834    ndcg@10 : 0.0607    hit@10 : 0.1647    precision@10 : 0.0223
  
  k:10, shrink:0.5
  Valid result:
  recall@10 : 0.0532    mrr@10 : 0.0531    ndcg@10 : 0.0386    hit@10 : 0.1132    precision@10 : 0.0146
  Test result:
  recall@10 : 0.0532    mrr@10 : 0.0525    ndcg@10 : 0.0384    hit@10 : 0.1127    precision@10 : 0.0145
  
  k:50, shrink:1.0
  Valid result:
  recall@10 : 0.071    mrr@10 : 0.07    ndcg@10 : 0.0512    hit@10 : 0.1452    precision@10 : 0.0193
  Test result:
  recall@10 : 0.0724    mrr@10 : 0.0697    ndcg@10 : 0.0515    hit@10 : 0.1455    precision@10 : 0.0194
  
  k:200, shrink:1.0
  Valid result:
  recall@10 : 0.0808    mrr@10 : 0.0803    ndcg@10 : 0.0587    hit@10 : 0.162    precision@10 : 0.0219
  Test result:
  recall@10 : 0.0828    mrr@10 : 0.081    ndcg@10 : 0.0598    hit@10 : 0.1633    precision@10 : 0.0221
  
  k:100, shrink:0.0
  Valid result:
  recall@10 : 0.0764    mrr@10 : 0.0754    ndcg@10 : 0.0551    hit@10 : 0.1543    precision@10 : 0.0207
  Test result:
  recall@10 : 0.0779    mrr@10 : 0.0759    ndcg@10 : 0.056    hit@10 : 0.1552    precision@10 : 0.0208
  
  k:100, shrink:0.5
  Valid result:
  recall@10 : 0.0769    mrr@10 : 0.0755    ndcg@10 : 0.0554    hit@10 : 0.1549    precision@10 : 0.0208
  Test result:
  recall@10 : 0.0784    mrr@10 : 0.0761    ndcg@10 : 0.0563    hit@10 : 0.1559    precision@10 : 0.0209
  
  k:200, shrink:0.5
  Valid result:
  recall@10 : 0.0806    mrr@10 : 0.0802    ndcg@10 : 0.0586    hit@10 : 0.1618    precision@10 : 0.0219
  Test result:
  recall@10 : 0.0824    mrr@10 : 0.0809    ndcg@10 : 0.0596    hit@10 : 0.1627    precision@10 : 0.0221
  
  k:400, shrink:1.0
  Valid result:
  recall@10 : 0.0829    mrr@10 : 0.0833    ndcg@10 : 0.0604    hit@10 : 0.1664    precision@10 : 0.0225
  Test result:
  recall@10 : 0.0842    mrr@10 : 0.0844    ndcg@10 : 0.0616    hit@10 : 0.1663    precision@10 : 0.0225
  
  k:250, shrink:0.0
  Valid result:
  recall@10 : 0.0811    mrr@10 : 0.0814    ndcg@10 : 0.0592    hit@10 : 0.1625    precision@10 : 0.022
  Test result:
  recall@10 : 0.0826    mrr@10 : 0.0822    ndcg@10 : 0.06    hit@10 : 0.1633    precision@10 : 0.0221
  
  k:400, shrink:0.5
  Valid result:
  recall@10 : 0.0824    mrr@10 : 0.0833    ndcg@10 : 0.0603    hit@10 : 0.1657    precision@10 : 0.0224
  Test result:
  recall@10 : 0.0838    mrr@10 : 0.0842    ndcg@10 : 0.0614    hit@10 : 0.1658    precision@10 : 0.0224
  
  k:10, shrink:1.0
  Valid result:
  recall@10 : 0.0535    mrr@10 : 0.0533    ndcg@10 : 0.0387    hit@10 : 0.1136    precision@10 : 0.0147
  Test result:
  recall@10 : 0.0538    mrr@10 : 0.0528    ndcg@10 : 0.0387    hit@10 : 0.1134    precision@10 : 0.0146
  
  k:100, shrink:1.0
  Valid result:
  recall@10 : 0.0771    mrr@10 : 0.0758    ndcg@10 : 0.0555    hit@10 : 0.1553    precision@10 : 0.0208
  Test result:
  recall@10 : 0.0787    mrr@10 : 0.0764    ndcg@10 : 0.0565    hit@10 : 0.1566    precision@10 : 0.021
  
  k:250, shrink:1.0
  Valid result:
  recall@10 : 0.0817    mrr@10 : 0.0818    ndcg@10 : 0.0596    hit@10 : 0.1634    precision@10 : 0.0222
  Test result:
  recall@10 : 0.0835    mrr@10 : 0.0824    ndcg@10 : 0.0605    hit@10 : 0.1648    precision@10 : 0.0223
  
  k:10, shrink:0.0
  Valid result:
  recall@10 : 0.0529    mrr@10 : 0.0528    ndcg@10 : 0.0383    hit@10 : 0.1129    precision@10 : 0.0146
  Test result:
  recall@10 : 0.0528    mrr@10 : 0.0522    ndcg@10 : 0.0381    hit@10 : 0.1121    precision@10 : 0.0144
  
  k:300, shrink:1.0
  Valid result:
  recall@10 : 0.082    mrr@10 : 0.0827    ndcg@10 : 0.06    hit@10 : 0.1649    precision@10 : 0.0223
  Test result:
  recall@10 : 0.0836    mrr@10 : 0.0837    ndcg@10 : 0.0612    hit@10 : 0.1656    precision@10 : 0.0224
  
  k:400, shrink:0.0
  Valid result:
  recall@10 : 0.082    mrr@10 : 0.0832    ndcg@10 : 0.0601    hit@10 : 0.1651    precision@10 : 0.0224
  Test result:
  recall@10 : 0.0835    mrr@10 : 0.0841    ndcg@10 : 0.0612    hit@10 : 0.1651    precision@10 : 0.0224
  
  k:200, shrink:0.0
  Valid result:
  recall@10 : 0.0803    mrr@10 : 0.0801    ndcg@10 : 0.0584    hit@10 : 0.1614    precision@10 : 0.0219
  Test result:
  recall@10 : 0.082    mrr@10 : 0.0807    ndcg@10 : 0.0593    hit@10 : 0.1622    precision@10 : 0.022
  
  k:50, shrink:0.5
  Valid result:
  recall@10 : 0.0705    mrr@10 : 0.0699    ndcg@10 : 0.051    hit@10 : 0.1444    precision@10 : 0.0192
  Test result:
  recall@10 : 0.0721    mrr@10 : 0.0694    ndcg@10 : 0.0513    hit@10 : 0.1449    precision@10 : 0.0193
  
  k:250, shrink:0.5
  Valid result:
  recall@10 : 0.0814    mrr@10 : 0.0817    ndcg@10 : 0.0595    hit@10 : 0.163    precision@10 : 0.0221
  Test result:
  recall@10 : 0.0829    mrr@10 : 0.0823    ndcg@10 : 0.0602    hit@10 : 0.1639    precision@10 : 0.0222
  ```
  
- **Logging Result**:

  ```yaml
  100%|██████████| 21/21 [6:10:53<00:00, 1059.68s/trial, best loss: -0.0604]
  best params:  {'k': 400, 'shrink': 1.0}
  best result: 
  {'model': 'ItemKNN', 'best_valid_score': 0.0604, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0829), ('mrr@10', 0.0833), ('ndcg@10', 0.0604), ('hit@10', 0.1664), ('precision@10', 0.0225)]), 'test_result': OrderedDict([('recall@10', 0.0842), ('mrr@10', 0.0844), ('ndcg@10', 0.0616), ('hit@10', 0.1663), ('precision@10', 0.0225)])}
  ```
