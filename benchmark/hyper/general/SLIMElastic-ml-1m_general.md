# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [SLIMElastic](https://recbole.io/docs/user_guide/model/general/slimelastic.html)

- **Time cost**: 1116.25s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  alpha choice [0.2,0.4,0.6,0.8] 
  l1_ratio choice [0.1,0.05,0.01,0.005] 
  ```

- **Best parameters**:

  ```yaml
  alpha: 0.2
  l1_ratio: 0.005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  alpha:0.6, l1_ratio:0.01
  Valid result:
  recall@10 : 0.1661    mrr@10 : 0.3803    ndcg@10 : 0.2094    hit@10 : 0.7119    precision@10 : 0.1512
  Test result:
  recall@10 : 0.1813    mrr@10 : 0.4498    ndcg@10 : 0.2512    hit@10 : 0.7345    precision@10 : 0.1809

  alpha:0.2, l1_ratio:0.005
  Valid result:
  recall@10 : 0.1831    mrr@10 : 0.3959    ndcg@10 : 0.226     hit@10 : 0.7434    precision@10 : 0.1645
  Test result:
  recall@10 : 0.2035    mrr@10 : 0.4781    ndcg@10 : 0.2782    hit@10 : 0.7638    precision@10 : 0.2015

  alpha:0.2, l1_ratio:0.05
  Valid result:
  recall@10 : 0.1661    mrr@10 : 0.3798    ndcg@10 : 0.2095    hit@10 : 0.7141    precision@10 : 0.1522
  Test result:
  recall@10 : 0.1823    mrr@10 : 0.447     ndcg@10 : 0.2508    hit@10 : 0.735     precision@10 : 0.1813
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 16/16 [4:57:40<00:00, 1116.25s/trial, best loss: -0.226]
  best params:  {'alpha': 0.2, 'l1_ratio': 0.005}
  best result: 
  {'model': 'SLIMElastic', 'best_valid_score': 0.226, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1831), ('mrr@10', 0.3959), ('ndcg@10', 0.226), ('hit@10', 0.7434), ('precision@10', 0.1645)]), 'test_result': OrderedDict([('recall@10', 0.2035), ('mrr@10', 0.4781), ('ndcg@10', 0.2782), ('hit@10', 0.7638), ('precision@10', 0.2015)])}
  ```
