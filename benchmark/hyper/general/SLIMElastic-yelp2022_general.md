# General Recommendation

- **Dataset**: [Yelp2022](../../md/yelp_general.md)

- **Model**: [SLIMElastic](https://recbole.io/docs/user_guide/model/general/slimelastic.html)

- **Time cost**: 3031.18s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  alpha choice [0.2,0.4,0.6,0.8]
  l1_ratio choice [0.1,0.05,0.01,0.005]
  hide_item choice [True]
  positive_only choice [True]
  ```
  
- **Best parameters**:

  ```yaml
  alpha: 0.2
  l1_ratio: 0.005
  hide_item: True
  positive_only: True
  ```
  
- **Hyper-parameter logging**:

  ```yaml
  alpha:0.2, hide_item:True, l1_ratio:0.05, positive_only:True
  Valid result:
  recall@10 : 0.0004    mrr@10 : 0.0004    ndcg@10 : 0.0003    hit@10 : 0.0013    precision@10 : 0.0001
  Test result:
  recall@10 : 0.0005    mrr@10 : 0.0003    ndcg@10 : 0.0003    hit@10 : 0.0012    precision@10 : 0.0001
  
  alpha:0.2, hide_item:True, l1_ratio:0.01, positive_only:True
  Valid result:
  recall@10 : 0.0099    mrr@10 : 0.0126    ndcg@10 : 0.0081    hit@10 : 0.0248    precision@10 : 0.0027
  Test result:
  recall@10 : 0.0105    mrr@10 : 0.0131    ndcg@10 : 0.0084    hit@10 : 0.0251    precision@10 : 0.0028
  
  alpha:0.8, hide_item:True, l1_ratio:0.01, positive_only:True
  Valid result:
  recall@10 : 0.0004    mrr@10 : 0.0004    ndcg@10 : 0.0003    hit@10 : 0.0013    precision@10 : 0.0001
  Test result:
  recall@10 : 0.0005    mrr@10 : 0.0003    ndcg@10 : 0.0003    hit@10 : 0.0012    precision@10 : 0.0001
  
  alpha:0.6, hide_item:True, l1_ratio:0.05, positive_only:True
  Valid result:
  recall@10 : 0.0004    mrr@10 : 0.0003    ndcg@10 : 0.0003    hit@10 : 0.0013    precision@10 : 0.0001
  Test result:
  recall@10 : 0.0005    mrr@10 : 0.0003    ndcg@10 : 0.0002    hit@10 : 0.0012    precision@10 : 0.0001
  
  alpha:0.2, hide_item:True, l1_ratio:0.005, positive_only:True
  Valid result:
  recall@10 : 0.0257    mrr@10 : 0.0249    ndcg@10 : 0.0184    hit@10 : 0.0579    precision@10 : 0.0065
  Test result:
  recall@10 : 0.0277    mrr@10 : 0.0257    ndcg@10 : 0.0193    hit@10 : 0.0593    precision@10 : 0.0067
  
  alpha:0.4, hide_item:True, l1_ratio:0.005, positive_only:True
  Valid result:
  recall@10 : 0.0099    mrr@10 : 0.0126    ndcg@10 : 0.0081    hit@10 : 0.0248    precision@10 : 0.0027
  Test result:
  recall@10 : 0.0106    mrr@10 : 0.0131    ndcg@10 : 0.0084    hit@10 : 0.0251    precision@10 : 0.0028
  
  alpha:0.2, hide_item:True, l1_ratio:0.1, positive_only:True
  Valid result:
  recall@10 : 0.0004    mrr@10 : 0.0003    ndcg@10 : 0.0003    hit@10 : 0.0013    precision@10 : 0.0001
  Test result:
  recall@10 : 0.0005    mrr@10 : 0.0003    ndcg@10 : 0.0002    hit@10 : 0.0012    precision@10 : 0.0001
  
  alpha:0.6, hide_item:True, l1_ratio:0.005, positive_only:True
  Valid result:
  recall@10 : 0.0027    mrr@10 : 0.0051    ndcg@10 : 0.0028    hit@10 : 0.0078    precision@10 : 0.0008
  Test result:
  recall@10 : 0.0025    mrr@10 : 0.005    ndcg@10 : 0.0025    hit@10 : 0.0074    precision@10 : 0.0008
  
  alpha:0.4, hide_item:True, l1_ratio:0.01, positive_only:True
  Valid result:
  recall@10 : 0.0005    mrr@10 : 0.0004    ndcg@10 : 0.0003    hit@10 : 0.0013    precision@10 : 0.0001
  Test result:
  recall@10 : 0.0005    mrr@10 : 0.0004    ndcg@10 : 0.0003    hit@10 : 0.0013    precision@10 : 0.0001
  
  alpha:0.8, hide_item:True, l1_ratio:0.005, positive_only:True
  Valid result:
  recall@10 : 0.0005    mrr@10 : 0.0004    ndcg@10 : 0.0003    hit@10 : 0.0013    precision@10 : 0.0001
  Test result:
  recall@10 : 0.0005    mrr@10 : 0.0004    ndcg@10 : 0.0003    hit@10 : 0.0013    precision@10 : 0.0001
  
  alpha:0.6, hide_item:True, l1_ratio:0.01, positive_only:True
  Valid result:
  recall@10 : 0.0005    mrr@10 : 0.0004    ndcg@10 : 0.0003    hit@10 : 0.0013    precision@10 : 0.0001
  Test result:
  recall@10 : 0.0005    mrr@10 : 0.0003    ndcg@10 : 0.0003    hit@10 : 0.0012    precision@10 : 0.0001
  
  alpha:0.8, hide_item:True, l1_ratio:0.1, positive_only:True
  Valid result:
  recall@10 : 0.0004    mrr@10 : 0.0003    ndcg@10 : 0.0003    hit@10 : 0.0013    precision@10 : 0.0001
  Test result:
  recall@10 : 0.0005    mrr@10 : 0.0003    ndcg@10 : 0.0002    hit@10 : 0.0012    precision@10 : 0.0001
  
  alpha:0.8, hide_item:True, l1_ratio:0.05, positive_only:True
  Valid result:
  recall@10 : 0.0004    mrr@10 : 0.0003    ndcg@10 : 0.0003    hit@10 : 0.0013    precision@10 : 0.0001
  Test result:
  recall@10 : 0.0005    mrr@10 : 0.0003    ndcg@10 : 0.0002    hit@10 : 0.0012    precision@10 : 0.0001
  
  alpha:0.6, hide_item:True, l1_ratio:0.1, positive_only:True
  Valid result:
  recall@10 : 0.0004    mrr@10 : 0.0003    ndcg@10 : 0.0003    hit@10 : 0.0013    precision@10 : 0.0001
  Test result:
  recall@10 : 0.0005    mrr@10 : 0.0003    ndcg@10 : 0.0002    hit@10 : 0.0012    precision@10 : 0.0001
  
  alpha:0.4, hide_item:True, l1_ratio:0.1, positive_only:True
  Valid result:
  recall@10 : 0.0004    mrr@10 : 0.0003    ndcg@10 : 0.0003    hit@10 : 0.0013    precision@10 : 0.0001
  Test result:
  recall@10 : 0.0005    mrr@10 : 0.0003    ndcg@10 : 0.0002    hit@10 : 0.0012    precision@10 : 0.0001
  
  alpha:0.4, hide_item:True, l1_ratio:0.05, positive_only:True
  Valid result:
  recall@10 : 0.0004    mrr@10 : 0.0003    ndcg@10 : 0.0003    hit@10 : 0.0013    precision@10 : 0.0001
  Test result:
  recall@10 : 0.0005    mrr@10 : 0.0003    ndcg@10 : 0.0002    hit@10 : 0.0012    precision@10 : 0.0001
  ```
  
- **Logging Result**:

  ```yaml
  100%|██████████| 16/16 [144:46:18<00:00, 32573.67s/trial, best loss: -0.0184]
  best params:  {'alpha': 0.2, 'hide_item': True, 'l1_ratio': 0.005, 'positive_only': True}
  best result: 
  {'model': 'SLIMElastic', 'best_valid_score': 0.0184, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0257), ('mrr@10', 0.0249), ('ndcg@10', 0.0184), ('hit@10', 0.0579), ('precision@10', 0.0065)]), 'test_result': OrderedDict([('recall@10', 0.0277), ('mrr@10', 0.0257), ('ndcg@10', 0.0193), ('hit@10', 0.0593), ('precision@10', 0.0067)])}
  ```
