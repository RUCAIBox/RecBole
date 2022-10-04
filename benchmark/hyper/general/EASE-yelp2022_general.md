# General Recommendation

- **Dataset**: [Yelp2022](../../md/yelp_general.md)

- **Model**: [EASE](https://recbole.io/docs/user_guide/model/general/ease.html)

- **Time cost**: 1637.30s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  reg_weight choice [1.0,10.0,100.0,500.0,1000.0,2000.0]
  ```

- **Best parameters**:

  ```yaml
   reg_weight: 10.0
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  reg_weight:2000.0
  Valid result:
  recall@10 : 0.08      mrr@10 : 0.0641    ndcg@10 : 0.0521    hit@10 : 0.1556    precision@10 : 0.0197
  Test result:
  recall@10 : 0.0841    mrr@10 : 0.0661    ndcg@10 : 0.0544    hit@10 : 0.1602    precision@10 : 0.0203
  
  reg_weight:1.0
  Valid result:
  recall@10 : 0.1055    mrr@10 : 0.1539    ndcg@10 : 0.1021    hit@10 : 0.1985    precision@10 : 0.0281
  Test result:
  recall@10 : 0.1072    mrr@10 : 0.1539    ndcg@10 : 0.1033    hit@10 : 0.2006    precision@10 : 0.0284
  
  reg_weight:10.0
  Valid result:
  recall@10 : 0.1176    mrr@10 : 0.1543    ndcg@10 : 0.1057    hit@10 : 0.2172    precision@10 : 0.0308
  Test result:
  recall@10 : 0.1196    mrr@10 : 0.1561    ndcg@10 : 0.1073    hit@10 : 0.2195    precision@10 : 0.0312
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 6/6 [2:43:43<00:00, 1637.30s/trial, best loss: -0.1057]
  best params:  {'reg_weight': 10.0}
  best result:
  {'model': 'EASE', 'best_valid_score': 0.1057, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1176), ('mrr@10', 0.1543), ('ndcg@10', 0.1057), ('hit@10', 0.2172), ('precision@10', 0.0308)]), 'test_result': OrderedDict([('recall@10', 0.1196), ('mrr@10', 0.1561), ('ndcg@10', 0.1073), ('hit@10', 0.2195), ('precision@10', 0.0312)])}
  ```