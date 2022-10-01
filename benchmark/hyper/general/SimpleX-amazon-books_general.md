# General Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_general.md)

- **Model**: [SimpleX](https://recbole.io/docs/user_guide/model/general/simplex.html)

- **Time cost**: 1166.29s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  gamma in [0.3,0.5,0.7] 
  margin in [0,0.5,0.9] 
  negative_weight in [1,10,50]
  ```

- **Best parameters**:

  ```yaml
  gamma: 0.5
  margin: 0.5
  negative_weight: 50
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  gamma:0.3, margin:0.5, negative_weight:1
  Valid result:
  recall@10 : 0.063    mrr@10 : 0.0407    ndcg@10 : 0.0377    hit@10 : 0.109    precision@10 : 0.0124
  Test result:
  recall@10 : 0.0605    mrr@10 : 0.0413    ndcg@10 : 0.0376    hit@10 : 0.1064    precision@10 : 0.0122

  gamma:0.7, margin:0, negative_weight:1
  Valid result:
  recall@10 : 0.0594    mrr@10 : 0.0418    ndcg@10 : 0.0376    hit@10 : 0.1043    precision@10 : 0.0121
  Test result:
  recall@10 : 0.061    mrr@10 : 0.0441    ndcg@10 : 0.0393    hit@10 : 0.1063    precision@10 : 0.0125

  gamma:0.7, margin:0, negative_weight:50
  Valid result:
  recall@10 : 0.1622    mrr@10 : 0.1241    ndcg@10 : 0.1099    hit@10 : 0.2659    precision@10 : 0.0331
  Test result:
  recall@10 : 0.1654    mrr@10 : 0.1328    ndcg@10 : 0.1157    hit@10 : 0.2685    precision@10 : 0.0346

  gamma:0.3, margin:0.9, negative_weight:1
  Valid result:
  recall@10 : 0.063    mrr@10 : 0.0425    ndcg@10 : 0.0385    hit@10 : 0.1131    precision@10 : 0.0129
  Test result:
  recall@10 : 0.0606    mrr@10 : 0.0439    ndcg@10 : 0.0383    hit@10 : 0.1093    precision@10 : 0.0128

  gamma:0.3, margin:0.5, negative_weight:50
  Valid result:
  recall@10 : 0.1775    mrr@10 : 0.1321    ndcg@10 : 0.1187    hit@10 : 0.2848    precision@10 : 0.0359
  Test result:
  recall@10 : 0.1817    mrr@10 : 0.1439    ndcg@10 : 0.1263    hit@10 : 0.2914    precision@10 : 0.0378

  gamma:0.3, margin:0, negative_weight:1
  Valid result:
  recall@10 : 0.0616    mrr@10 : 0.0424    ndcg@10 : 0.0388    hit@10 : 0.1075    precision@10 : 0.0124
  Test result:
  recall@10 : 0.061    mrr@10 : 0.0432    ndcg@10 : 0.039    hit@10 : 0.1062    precision@10 : 0.0123

  gamma:0.7, margin:0.5, negative_weight:10
  Valid result:
  recall@10 : 0.1321    mrr@10 : 0.0923    ndcg@10 : 0.0847    hit@10 : 0.217    precision@10 : 0.0259
  Test result:
  recall@10 : 0.1334    mrr@10 : 0.0988    ndcg@10 : 0.0889    hit@10 : 0.2186    precision@10 : 0.0269

  gamma:0.7, margin:0.9, negative_weight:10
  Valid result:
  recall@10 : 0.0974    mrr@10 : 0.0669    ndcg@10 : 0.0613    hit@10 : 0.1652    precision@10 : 0.0194
  Test result:
  recall@10 : 0.0969    mrr@10 : 0.0696    ndcg@10 : 0.0623    hit@10 : 0.1651    precision@10 : 0.0198

  gamma:0.5, margin:0.5, negative_weight:1
  Valid result:
  recall@10 : 0.0645    mrr@10 : 0.0423    ndcg@10 : 0.0391    hit@10 : 0.1107    precision@10 : 0.0126
  Test result:
  recall@10 : 0.0635    mrr@10 : 0.0432    ndcg@10 : 0.0394    hit@10 : 0.1106    precision@10 : 0.0127

  gamma:0.5, margin:0.9, negative_weight:10
  Valid result:
  recall@10 : 0.1092    mrr@10 : 0.0746    ndcg@10 : 0.0684    hit@10 : 0.184    precision@10 : 0.0217
  Test result:
  recall@10 : 0.1119    mrr@10 : 0.0787    ndcg@10 : 0.0716    hit@10 : 0.1878    precision@10 : 0.0225

  gamma:0.5, margin:0.5, negative_weight:50
  Valid result:
  recall@10 : 0.1841    mrr@10 : 0.1389    ndcg@10 : 0.1241    hit@10 : 0.2966    precision@10 : 0.0373
  Test result:
  recall@10 : 0.1896    mrr@10 : 0.1517    ndcg@10 : 0.133    hit@10 : 0.3023    precision@10 : 0.0398

  gamma:0.5, margin:0.9, negative_weight:1
  Valid result:
  recall@10 : 0.0663    mrr@10 : 0.0438    ndcg@10 : 0.0402    hit@10 : 0.1167    precision@10 : 0.0133
  Test result:
  recall@10 : 0.0627    mrr@10 : 0.0447    ndcg@10 : 0.0396    hit@10 : 0.1119    precision@10 : 0.0132

  gamma:0.7, margin:0.9, negative_weight:1
  Valid result:
  recall@10 : 0.0641    mrr@10 : 0.0429    ndcg@10 : 0.039    hit@10 : 0.113    precision@10 : 0.0128
  Test result:
  recall@10 : 0.0604    mrr@10 : 0.0426    ndcg@10 : 0.0378    hit@10 : 0.1087    precision@10 : 0.0126

  gamma:0.7, margin:0.5, negative_weight:50
  Valid result:
  recall@10 : 0.1771    mrr@10 : 0.1325    ndcg@10 : 0.119    hit@10 : 0.2867    precision@10 : 0.0362
  Test result:
  recall@10 : 0.1817    mrr@10 : 0.1447    ndcg@10 : 0.127    hit@10 : 0.2913    precision@10 : 0.038

  gamma:0.7, margin:0.5, negative_weight:1
  Valid result:
  recall@10 : 0.0648    mrr@10 : 0.0424    ndcg@10 : 0.0394    hit@10 : 0.1117    precision@10 : 0.0127
  Test result:
  recall@10 : 0.0639    mrr@10 : 0.0434    ndcg@10 : 0.0397    hit@10 : 0.1114    precision@10 : 0.0129

  gamma:0.5, margin:0, negative_weight:1
  Valid result:
  recall@10 : 0.0588    mrr@10 : 0.0405    ndcg@10 : 0.0366    hit@10 : 0.1034    precision@10 : 0.012
  Test result:
  recall@10 : 0.0586    mrr@10 : 0.0405    ndcg@10 : 0.0368    hit@10 : 0.1029    precision@10 : 0.0119

  gamma:0.3, margin:0.9, negative_weight:50
  Valid result:
  recall@10 : 0.1065    mrr@10 : 0.0759    ndcg@10 : 0.0681    hit@10 : 0.1819    precision@10 : 0.0216
  Test result:
  recall@10 : 0.1081    mrr@10 : 0.0797    ndcg@10 : 0.0708    hit@10 : 0.1852    precision@10 : 0.0226

  gamma:0.7, margin:0.9, negative_weight:50
  Valid result:
  recall@10 : 0.1246    mrr@10 : 0.0878    ndcg@10 : 0.0797    hit@10 : 0.2094    precision@10 : 0.0252
  Test result:
  recall@10 : 0.1265    mrr@10 : 0.0943    ndcg@10 : 0.0837    hit@10 : 0.212    precision@10 : 0.0264

  gamma:0.5, margin:0.9, negative_weight:50
  Valid result:
  recall@10 : 0.1266    mrr@10 : 0.0904    ndcg@10 : 0.0818    hit@10 : 0.2105    precision@10 : 0.0254
  Test result:
  recall@10 : 0.1283    mrr@10 : 0.0966    ndcg@10 : 0.0853    hit@10 : 0.2143    precision@10 : 0.0265

  gamma:0.3, margin:0.9, negative_weight:10
  Valid result:
  recall@10 : 0.0866    mrr@10 : 0.0579    ndcg@10 : 0.0531    hit@10 : 0.1494    precision@10 : 0.0172
  Test result:
  recall@10 : 0.0867    mrr@10 : 0.0607    ndcg@10 : 0.0548    hit@10 : 0.1497    precision@10 : 0.0177

  gamma:0.5, margin:0, negative_weight:50
  Valid result:
  recall@10 : 0.159    mrr@10 : 0.1191    ndcg@10 : 0.1062    hit@10 : 0.2602    precision@10 : 0.0321
  Test result:
  recall@10 : 0.1617    mrr@10 : 0.1275    ndcg@10 : 0.112    hit@10 : 0.2622    precision@10 : 0.0334
  ```

- **Logging Result**:

  ```yaml
  INFO  Early stop triggered. Stopping iterations as condition is reach.
  78%|███████▊  | 21/27 [6:48:12<1:56:37, 1166.29s/trial, best loss: -0.1241]
  best params:  {'gamma': 0.5, 'margin': 0.5, 'negative_weight': 50}
  best result: 
  {'model': 'SimpleX', 'best_valid_score': 0.1241, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1841), ('mrr@10', 0.1389), ('ndcg@10', 0.1241), ('hit@10', 0.2966), ('precision@10', 0.0373)]), 'test_result': OrderedDict([('recall@10', 0.1896), ('mrr@10', 0.1517), ('ndcg@10', 0.133), ('hit@10', 0.3023), ('precision@10', 0.0398)])}
  ```
