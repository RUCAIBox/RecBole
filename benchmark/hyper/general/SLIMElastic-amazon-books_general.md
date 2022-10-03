# General Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_general.md)

- **Model**: [SLIMElastic](https://recbole.io/docs/user_guide/model/general/slimelastic.html)

- **Time cost**: 10937.82s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  alpha in [0.2,0.4,0.6,0.8] 
  l1_ratio in [0.1,0.05,0.01,0.005]
  ```

- **Best parameters**:

  ```yaml
  alpha: 0.2
  l1_ratio: 0.005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  alpha:0.2, l1_ratio:0.1
  Valid result:
  recall@10 : 0.0048    mrr@10 : 0.0055    ndcg@10 : 0.0048    hit@10 : 0.0059    precision@10 : 0.0006
  Test result:
  recall@10 : 0.0037    mrr@10 : 0.0044    ndcg@10 : 0.0037    hit@10 : 0.0047    precision@10 : 0.0005

  alpha:0.6, l1_ratio:0.1
  Valid result:
  recall@10 : 0.0004    mrr@10 : 0.0001    ndcg@10 : 0.0002    hit@10 : 0.0005    precision@10 : 0.0001
  Test result:
  recall@10 : 0.0003    mrr@10 : 0.0    ndcg@10 : 0.0001    hit@10 : 0.0003    precision@10 : 0.0

  alpha:0.8, l1_ratio:0.01
  Valid result:
  recall@10 : 0.0049    mrr@10 : 0.0055    ndcg@10 : 0.0048    hit@10 : 0.006    precision@10 : 0.0006
  Test result:
  recall@10 : 0.0037    mrr@10 : 0.0044    ndcg@10 : 0.0037    hit@10 : 0.0047    precision@10 : 0.0005

  alpha:0.4, l1_ratio:0.05
  Valid result:
  recall@10 : 0.0048    mrr@10 : 0.0055    ndcg@10 : 0.0048    hit@10 : 0.0059    precision@10 : 0.0006
  Test result:
  recall@10 : 0.0037    mrr@10 : 0.0044    ndcg@10 : 0.0037    hit@10 : 0.0047    precision@10 : 0.0005

  alpha:0.8, l1_ratio:0.1
  Valid result:
  recall@10 : 0.0004    mrr@10 : 0.0001    ndcg@10 : 0.0002    hit@10 : 0.0005    precision@10 : 0.0001
  Test result:
  recall@10 : 0.0003    mrr@10 : 0.0    ndcg@10 : 0.0001    hit@10 : 0.0003    precision@10 : 0.0

  alpha:0.2, l1_ratio:0.005
  Valid result:
  recall@10 : 0.0922    mrr@10 : 0.0963    ndcg@10 : 0.0743    hit@10 : 0.1712    precision@10 : 0.0217
  Test result:
  recall@10 : 0.0942    mrr@10 : 0.106    ndcg@10 : 0.0795    hit@10 : 0.1748    precision@10 : 0.0228

  alpha:0.6, l1_ratio:0.005
  Valid result:
  recall@10 : 0.0215    mrr@10 : 0.0344    ndcg@10 : 0.0216    hit@10 : 0.0534    precision@10 : 0.0069
  Test result:
  recall@10 : 0.0205    mrr@10 : 0.0352    ndcg@10 : 0.0213    hit@10 : 0.0527    precision@10 : 0.0069

  alpha:0.6, l1_ratio:0.05
  Valid result:
  recall@10 : 0.0004    mrr@10 : 0.0001    ndcg@10 : 0.0002    hit@10 : 0.0005    precision@10 : 0.0001
  Test result:
  recall@10 : 0.0003    mrr@10 : 0.0    ndcg@10 : 0.0001    hit@10 : 0.0003    precision@10 : 0.0

  alpha:0.4, l1_ratio:0.01
  Valid result:
  recall@10 : 0.0099    mrr@10 : 0.0192    ndcg@10 : 0.011    hit@10 : 0.0246    precision@10 : 0.0028
  Test result:
  recall@10 : 0.0087    mrr@10 : 0.0192    ndcg@10 : 0.0102    hit@10 : 0.0236    precision@10 : 0.0027

  alpha:0.6, l1_ratio:0.01
  Valid result:
  recall@10 : 0.0063    mrr@10 : 0.0109    ndcg@10 : 0.0069    hit@10 : 0.0118    precision@10 : 0.0012
  Test result:
  recall@10 : 0.0051    mrr@10 : 0.0092    ndcg@10 : 0.0056    hit@10 : 0.0099    precision@10 : 0.001

  alpha:0.8, l1_ratio:0.05
  Valid result:
  recall@10 : 0.0004    mrr@10 : 0.0001    ndcg@10 : 0.0002    hit@10 : 0.0005    precision@10 : 0.0001
  Test result:
  recall@10 : 0.0003    mrr@10 : 0.0    ndcg@10 : 0.0001    hit@10 : 0.0003    precision@10 : 0.0

  alpha:0.8, l1_ratio:0.005
  Valid result:
  recall@10 : 0.0099    mrr@10 : 0.0192    ndcg@10 : 0.011    hit@10 : 0.0246    precision@10 : 0.0028
  Test result:
  recall@10 : 0.0087    mrr@10 : 0.0192    ndcg@10 : 0.0102    hit@10 : 0.0236    precision@10 : 0.0027

  alpha:0.4, l1_ratio:0.005
  Valid result:
  recall@10 : 0.0438    mrr@10 : 0.057    ndcg@10 : 0.0395    hit@10 : 0.0933    precision@10 : 0.012
  Test result:
  recall@10 : 0.0443    mrr@10 : 0.0615    ndcg@10 : 0.0416    hit@10 : 0.0957    precision@10 : 0.0126

  alpha:0.4, l1_ratio:0.1
  Valid result:
  recall@10 : 0.0004    mrr@10 : 0.0001    ndcg@10 : 0.0002    hit@10 : 0.0005    precision@10 : 0.0001
  Test result:
  recall@10 : 0.0003    mrr@10 : 0.0    ndcg@10 : 0.0001    hit@10 : 0.0003    precision@10 : 0.0

  alpha:0.2, l1_ratio:0.01
  Valid result:
  recall@10 : 0.044    mrr@10 : 0.0577    ndcg@10 : 0.0398    hit@10 : 0.0937    precision@10 : 0.0121
  Test result:
  recall@10 : 0.0445    mrr@10 : 0.0622    ndcg@10 : 0.0418    hit@10 : 0.0959    precision@10 : 0.0127

  alpha:0.2, l1_ratio:0.05
  Valid result:
  recall@10 : 0.0048    mrr@10 : 0.0055    ndcg@10 : 0.0048    hit@10 : 0.0059    precision@10 : 0.0006
  Test result:
  recall@10 : 0.0037    mrr@10 : 0.0044    ndcg@10 : 0.0037    hit@10 : 0.0047    precision@10 : 0.0005
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 16/16 [48:36:45<00:00, 10937.82s/trial, best loss: -0.0743]
  best params:  {'alpha': 0.2, 'l1_ratio': 0.005}
  best result: 
  {'model': 'SLIMElastic', 'best_valid_score': 0.0743, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0922), ('mrr@10', 0.0963), ('ndcg@10', 0.0743), ('hit@10', 0.1712), ('precision@10', 0.0217)]), 'test_result': OrderedDict([('recall@10', 0.0942), ('mrr@10', 0.106), ('ndcg@10', 0.0795), ('hit@10', 0.1748), ('precision@10', 0.0228)])}
  ```
