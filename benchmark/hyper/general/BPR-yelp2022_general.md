# General Recommendation

- **Dataset**: [Yelp2022](../../md/yelp_general.md)

- **Model**: [BPR](https://recbole.io/docs/user_guide/model/general/bpr.html)

- **Time cost**: 4051.46s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [5e-5,1e-4,5e-4,7e-4,1e-3,3e-3,5e-3]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 1e-3
  ```

- **Hyper-parameter logging**:

  ```yaml
  learning_rate:0.005
  Valid result:
  recall@10 : 0.0505    mrr@10 : 0.0369    ndcg@10 : 0.0313    hit@10 : 0.0975    precision@10 : 0.0109
  Test result:
  recall@10 : 0.0505    mrr@10 : 0.0365    ndcg@10 : 0.0311    hit@10 : 0.0979    precision@10 : 0.0109
  
  learning_rate:0.001
  Valid result:
  recall@10 : 0.0761    mrr@10 : 0.0543    ndcg@10 : 0.0473    hit@10 : 0.1397    precision@10 : 0.0164
  Test result:
  recall@10 : 0.0756    mrr@10 : 0.0549    ndcg@10 : 0.0475    hit@10 : 0.1389    precision@10 : 0.0163
  
  learning_rate:0.0001
  Valid result:
  recall@10 : 0.0592    mrr@10 : 0.0438    ndcg@10 : 0.0369    hit@10 : 0.1157    precision@10 : 0.0132
  Test result:
  recall@10 : 0.0597    mrr@10 : 0.044    ndcg@10 : 0.037    hit@10 : 0.1158    precision@10 : 0.0133
  
  learning_rate:5e-05
  Valid result:
  recall@10 : 0.0383    mrr@10 : 0.0304    ndcg@10 : 0.0246    hit@10 : 0.0785    precision@10 : 0.0087
  Test result:
  recall@10 : 0.039    mrr@10 : 0.0304    ndcg@10 : 0.0248    hit@10 : 0.0784    precision@10 : 0.0088
  
  learning_rate:0.0005
  Valid result:
  recall@10 : 0.0744    mrr@10 : 0.0542    ndcg@10 : 0.0464    hit@10 : 0.1396    precision@10 : 0.0164
  Test result:
  recall@10 : 0.0753    mrr@10 : 0.0552    ndcg@10 : 0.047    hit@10 : 0.1408    precision@10 : 0.0166
  
  learning_rate:0.0007
  Valid result:
  recall@10 : 0.0726    mrr@10 : 0.0532    ndcg@10 : 0.0455    hit@10 : 0.1369    precision@10 : 0.016
  Test result:
  recall@10 : 0.0733    mrr@10 : 0.0541    ndcg@10 : 0.0461    hit@10 : 0.1378    precision@10 : 0.0162
  
  learning_rate:0.003
  Valid result:
  recall@10 : 0.0618    mrr@10 : 0.044    ndcg@10 : 0.0381    hit@10 : 0.1169    precision@10 : 0.0132
  Test result:
  recall@10 : 0.0619    mrr@10 : 0.0442    ndcg@10 : 0.0383    hit@10 : 0.1164    precision@10 : 0.0133
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 7/7 [7:52:40<00:00, 4051.46s/trial, best loss: -0.0473]
  best params:  {'learning_rate': 0.001}
  best result: 
  {'model': 'BPR', 'best_valid_score': 0.0473, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0761), ('mrr@10', 0.0543), ('ndcg@10', 0.0473), ('hit@10', 0.1397), ('precision@10', 0.0164)]), 'test_result': OrderedDict([('recall@10', 0.0756), ('mrr@10', 0.0549), ('ndcg@10', 0.0475), ('hit@10', 0.1389), ('precision@10', 0.0163)])}
  ```
  
  
