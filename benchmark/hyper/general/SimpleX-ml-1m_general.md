# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [SimpleX](https://recbole.io/docs/user_guide/model/general/simplex.html)

- **Time cost**: 4020.75s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  gamma choice [0.3,0.5,0.7] 
  margin choice [0,0.5,0.9] 
  negative_weight choice [1,10,50] 
  sample_num choice [50]
  ```

- **Best parameters**:

  ```yaml
  gamma: 0.7  
  margin: 0.9  
  negative_weight: 50
  sample_num: 50
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  gamma:0.5, margin:0.5, negative_weight:1, sample_num:50
  Valid result:
  recall@10 : 0.0698    mrr@10 : 0.1883    ndcg@10 : 0.0916    hit@10 : 0.4504    precision@10 : 0.0729
  Test result:
  recall@10 : 0.0709    mrr@10 : 0.2028    ndcg@10 : 0.1003    hit@10 : 0.4616    precision@10 : 0.0805

  gamma:0.7, margin:0, negative_weight:10, sample_num:50
  Valid result:
  recall@10 : 0.1529    mrr@10 : 0.3501    ndcg@10 : 0.1926    hit@10 : 0.6856    precision@10 : 0.1445
  Test result:
  recall@10 : 0.1663    mrr@10 : 0.4177    ndcg@10 : 0.2335    hit@10 : 0.7055    precision@10 : 0.1724

  gamma:0.7, margin:0.9, negative_weight:50, sample_num:50
  Valid result:
  recall@10 : 0.1822    mrr@10 : 0.3979    ndcg@10 : 0.2249    hit@10 : 0.7396    precision@10 : 0.1627
  Test result:
  recall@10 : 0.1997    mrr@10 : 0.4668    ndcg@10 : 0.27      hit@10 : 0.7555    precision@10 : 0.1948
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 27/27 [30:09:20<00:00, 4020.75s/trial, best loss: -0.2249]
  best params:  {'gamma': 0.7, 'margin': 0.9, 'negative_weight': 50, 'sample_num': 50}
  best result: 
  {'model': 'SimpleX', 'best_valid_score': 0.2249, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1822), ('mrr@10', 0.3979), ('ndcg@10', 0.2249), ('hit@10', 0.7396), ('precision@10', 0.1627)]), 'test_result': OrderedDict([('recall@10', 0.1997), ('mrr@10', 0.4668), ('ndcg@10', 0.27), ('hit@10', 0.7555), ('precision@10', 0.1948)])}
  ```
