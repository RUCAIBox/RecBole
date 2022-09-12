# General Recommendation

- **Dataset**: [Yelp2022](../../md/yelp_general.md)

- **Model**: [NCEPLRec](https://recbole.io/docs/user_guide/model/general/nceplrec.html)

- **Time cost**: 3031.18s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  rank choice [100,200,450]
  beta choice [1.0,1.3]
  reg_weight choice [1e-04,1e-02,1e2,15000]
  ```
  
- **Best parameters**:

  ```yaml
  rank: 450
  beta: 1.3
  reg_weight: 1e2
  ```
  
- **Hyper-parameter logging**:

  ```yaml
  beta:1.0, rank:450, reg_weight:15000
  Valid result:
  recall@10 : 0.0817    mrr@10 : 0.0709    ndcg@10 : 0.0558    hit@10 : 0.1576    precision@10 : 0.02
  Test result:
  recall@10 : 0.0837    mrr@10 : 0.0743    ndcg@10 : 0.058    hit@10 : 0.1612    precision@10 : 0.0206
  
  beta:1.3, rank:100, reg_weight:15000
  Valid result:
  recall@10 : 0.0653    mrr@10 : 0.0504    ndcg@10 : 0.0415    hit@10 : 0.1285    precision@10 : 0.0153
  Test result:
  recall@10 : 0.0671    mrr@10 : 0.0526    ndcg@10 : 0.0431    hit@10 : 0.1305    precision@10 : 0.0156
  
  beta:1.3, rank:200, reg_weight:0.0001
  Valid result:
  recall@10 : 0.0764    mrr@10 : 0.0597    ndcg@10 : 0.0493    hit@10 : 0.1472    precision@10 : 0.0181
  Test result:
  recall@10 : 0.0789    mrr@10 : 0.0615    ndcg@10 : 0.0508    hit@10 : 0.1503    precision@10 : 0.0185
  
  beta:1.0, rank:450, reg_weight:0.0001
  Valid result:
  recall@10 : 0.0812    mrr@10 : 0.0715    ndcg@10 : 0.0559    hit@10 : 0.1571    precision@10 : 0.02
  Test result:
  recall@10 : 0.0832    mrr@10 : 0.0746    ndcg@10 : 0.0579    hit@10 : 0.1607    precision@10 : 0.0205
  
  beta:1.0, rank:450, reg_weight:0.01
  Valid result:
  recall@10 : 0.0812    mrr@10 : 0.0715    ndcg@10 : 0.0559    hit@10 : 0.1571    precision@10 : 0.02
  Test result:
  recall@10 : 0.0832    mrr@10 : 0.0746    ndcg@10 : 0.0579    hit@10 : 0.1607    precision@10 : 0.0205
  
  beta:1.3, rank:450, reg_weight:100.0
  Valid result:
  recall@10 : 0.0883    mrr@10 : 0.0728    ndcg@10 : 0.0594    hit@10 : 0.1668    precision@10 : 0.0213
  Test result:
  recall@10 : 0.0908    mrr@10 : 0.0752    ndcg@10 : 0.0612    hit@10 : 0.1702    precision@10 : 0.0218
  
  beta:1.0, rank:200, reg_weight:100.0
  Valid result:
  recall@10 : 0.0738    mrr@10 : 0.0603    ndcg@10 : 0.0486    hit@10 : 0.144    precision@10 : 0.0176
  Test result:
  recall@10 : 0.0769    mrr@10 : 0.0618    ndcg@10 : 0.05    hit@10 : 0.1476    precision@10 : 0.018
  
  beta:1.3, rank:450, reg_weight:0.0001
  Valid result:
  recall@10 : 0.0884    mrr@10 : 0.0728    ndcg@10 : 0.0594    hit@10 : 0.1669    precision@10 : 0.0213
  Test result:
  recall@10 : 0.0908    mrr@10 : 0.0752    ndcg@10 : 0.0612    hit@10 : 0.1701    precision@10 : 0.0218
  
  beta:1.0, rank:200, reg_weight:0.0001
  Valid result:
  recall@10 : 0.0738    mrr@10 : 0.0603    ndcg@10 : 0.0486    hit@10 : 0.144    precision@10 : 0.0176
  Test result:
  recall@10 : 0.0769    mrr@10 : 0.0618    ndcg@10 : 0.0501    hit@10 : 0.1476    precision@10 : 0.0181
  
  beta:1.0, rank:200, reg_weight:15000
  Valid result:
  recall@10 : 0.0741    mrr@10 : 0.06    ndcg@10 : 0.0486    hit@10 : 0.144    precision@10 : 0.0176
  Test result:
  recall@10 : 0.0768    mrr@10 : 0.0615    ndcg@10 : 0.0499    hit@10 : 0.1472    precision@10 : 0.018
  
  beta:1.3, rank:200, reg_weight:100.0
  Valid result:
  recall@10 : 0.0764    mrr@10 : 0.0597    ndcg@10 : 0.0493    hit@10 : 0.1472    precision@10 : 0.0181
  Test result:
  recall@10 : 0.0789    mrr@10 : 0.0615    ndcg@10 : 0.0508    hit@10 : 0.1503    precision@10 : 0.0185
  
  beta:1.3, rank:200, reg_weight:15000
  Valid result:
  recall@10 : 0.0752    mrr@10 : 0.0589    ndcg@10 : 0.0487    hit@10 : 0.1455    precision@10 : 0.0178
  Test result:
  recall@10 : 0.0782    mrr@10 : 0.0606    ndcg@10 : 0.0502    hit@10 : 0.1486    precision@10 : 0.0183
  
  beta:1.0, rank:100, reg_weight:100.0
  Valid result:
  recall@10 : 0.0662    mrr@10 : 0.0519    ndcg@10 : 0.0424    hit@10 : 0.13    precision@10 : 0.0155
  Test result:
  recall@10 : 0.0686    mrr@10 : 0.0533    ndcg@10 : 0.0438    hit@10 : 0.1332    precision@10 : 0.0159
  
  beta:1.3, rank:100, reg_weight:100.0
  Valid result:
  recall@10 : 0.0658    mrr@10 : 0.0508    ndcg@10 : 0.0419    hit@10 : 0.1297    precision@10 : 0.0155
  Test result:
  recall@10 : 0.0676    mrr@10 : 0.0529    ndcg@10 : 0.0434    hit@10 : 0.1313    precision@10 : 0.0157
  
  beta:1.0, rank:100, reg_weight:15000
  Valid result:
  recall@10 : 0.0658    mrr@10 : 0.0516    ndcg@10 : 0.0422    hit@10 : 0.1292    precision@10 : 0.0154
  Test result:
  recall@10 : 0.0682    mrr@10 : 0.0532    ndcg@10 : 0.0436    hit@10 : 0.1325    precision@10 : 0.0158
  
  beta:1.3, rank:450, reg_weight:0.01
  Valid result:
  recall@10 : 0.0884    mrr@10 : 0.0728    ndcg@10 : 0.0594    hit@10 : 0.1669    precision@10 : 0.0213
  Test result:
  recall@10 : 0.0908    mrr@10 : 0.0752    ndcg@10 : 0.0612    hit@10 : 0.1701    precision@10 : 0.0218
  
  beta:1.0, rank:100, reg_weight:0.01
  Valid result:
  recall@10 : 0.0662    mrr@10 : 0.0519    ndcg@10 : 0.0425    hit@10 : 0.13    precision@10 : 0.0155
  Test result:
  recall@10 : 0.0686    mrr@10 : 0.0533    ndcg@10 : 0.0438    hit@10 : 0.1332    precision@10 : 0.0159
  
  beta:1.3, rank:200, reg_weight:0.01
  Valid result:
  recall@10 : 0.0764    mrr@10 : 0.0597    ndcg@10 : 0.0493    hit@10 : 0.1472    precision@10 : 0.0181
  Test result:
  recall@10 : 0.0789    mrr@10 : 0.0615    ndcg@10 : 0.0508    hit@10 : 0.1503    precision@10 : 0.0185
  
  beta:1.0, rank:450, reg_weight:100.0
  Valid result:
  recall@10 : 0.0813    mrr@10 : 0.0715    ndcg@10 : 0.0559    hit@10 : 0.1571    precision@10 : 0.02
  Test result:
  recall@10 : 0.0832    mrr@10 : 0.0746    ndcg@10 : 0.0579    hit@10 : 0.1607    precision@10 : 0.0205
  
  beta:1.3, rank:100, reg_weight:0.0001
  Valid result:
  recall@10 : 0.0658    mrr@10 : 0.0508    ndcg@10 : 0.0419    hit@10 : 0.1297    precision@10 : 0.0155
  Test result:
  recall@10 : 0.0676    mrr@10 : 0.053    ndcg@10 : 0.0434    hit@10 : 0.1313    precision@10 : 0.0157
  
  beta:1.0, rank:100, reg_weight:0.0001
  Valid result:
  recall@10 : 0.0662    mrr@10 : 0.0519    ndcg@10 : 0.0425    hit@10 : 0.13    precision@10 : 0.0155
  Test result:
  recall@10 : 0.0686    mrr@10 : 0.0533    ndcg@10 : 0.0438    hit@10 : 0.1332    precision@10 : 0.0159
  
  beta:1.3, rank:100, reg_weight:0.01
  Valid result:
  recall@10 : 0.0658    mrr@10 : 0.0508    ndcg@10 : 0.0419    hit@10 : 0.1297    precision@10 : 0.0155
  Test result:
  recall@10 : 0.0676    mrr@10 : 0.053    ndcg@10 : 0.0434    hit@10 : 0.1313    precision@10 : 0.0157
  
  beta:1.3, rank:450, reg_weight:15000
  Valid result:
  recall@10 : 0.0869    mrr@10 : 0.0707    ndcg@10 : 0.0579    hit@10 : 0.1645    precision@10 : 0.021
  Test result:
  recall@10 : 0.0893    mrr@10 : 0.0732    ndcg@10 : 0.0598    hit@10 : 0.1673    precision@10 : 0.0214
  
  beta:1.0, rank:200, reg_weight:0.01
  Valid result:
  recall@10 : 0.0738    mrr@10 : 0.0603    ndcg@10 : 0.0486    hit@10 : 0.144    precision@10 : 0.0176
  Test result:
  recall@10 : 0.0769    mrr@10 : 0.0618    ndcg@10 : 0.0501    hit@10 : 0.1476    precision@10 : 0.0181
  ```
  
- **Logging Result**:

  ```yaml
  100%|██████████| 24/24 [92:58:03<00:00, 13945.15s/trial, best loss: -0.0594]
  best params:  {'beta': 1.3, 'rank': 450, 'reg_weight': 100.0}
  best result: 
  {'model': 'NCEPLRec', 'best_valid_score': 0.0594, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0883), ('mrr@10', 0.0728), ('ndcg@10', 0.0594), ('hit@10', 0.1668), ('precision@10', 0.0213)]), 'test_result': OrderedDict([('recall@10', 0.0908), ('mrr@10', 0.0752), ('ndcg@10', 0.0612), ('hit@10', 0.1702), ('precision@10', 0.0218)])}
  ```
