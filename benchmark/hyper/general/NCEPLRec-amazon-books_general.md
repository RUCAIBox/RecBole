# General Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_general.md)

- **Model**: [NCEPLRec](https://recbole.io/docs/user_guide/model/general/nceplrec.html)

- **Time cost**: 4059.08s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  rank choice [100,200,450] 
  beta choice [0.8,1.0,1.3] 
  reg_weight choice [1e-4,1e-2,1e2,15000]
  ```

- **Best parameters**:

  ```yaml
  rank: 450  
  beta: 1.0  
  reg_weight: 100.0
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  beta:1.0, rank:450, reg_weight:100.0
  Valid result:
  recall@10 : 0.1763    mrr@10 : 0.1493    ndcg@10 : 0.1258    hit@10 : 0.2977    precision@10 : 0.0416
  Test result:
  recall@10 : 0.1827    mrr@10 : 0.1671    ndcg@10 : 0.1375    hit@10 : 0.3029    precision@10 : 0.0448
  
  beta:1.3, rank:100, reg_weight:0.01
  Valid result:
  recall@10 : 0.1041    mrr@10 : 0.0919    ndcg@10 : 0.0735    hit@10 : 0.1958    precision@10 : 0.0263
  Test result:
  recall@10 : 0.1082    mrr@10 : 0.1018    ndcg@10 : 0.0801    hit@10 : 0.2001    precision@10 : 0.028

  beta:1.3, rank:100, reg_weight:100.0
  Valid result:
  recall@10 : 0.1041    mrr@10 : 0.0918    ndcg@10 : 0.0734    hit@10 : 0.1958    precision@10 : 0.0263
  Test result:
  recall@10 : 0.1082    mrr@10 : 0.1018    ndcg@10 : 0.0801    hit@10 : 0.2001    precision@10 : 0.028
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 36/36 [40:35:26<00:00, 4059.08s/trial, best loss: -0.1258]
  best params:  {'beta': 1.0, 'rank': 450, 'reg_weight': 100.0}
  best result:
  {'model': 'NCEPLRec', 'best_valid_score': 0.1258, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1763), ('mrr@10', 0.1493), ('ndcg@10', 0.1258), ('hit@10', 0.2977), ('precision@10', 0.0416)]), 'test_result': OrderedDict([('recall@10', 0.1827), ('mrr@10', 0.1671), ('ndcg@10', 0.1375), ('hit@10', 0.3029), ('precision@10', 0.0448)])}
  ```
