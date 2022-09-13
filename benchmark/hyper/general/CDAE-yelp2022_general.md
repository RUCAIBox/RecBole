# General Recommendation

- **Dataset**: [Yelp2022](../../md/yelp_general.md)

- **Model**: [CDAE](https://recbole.io/docs/user_guide/model/general/cdae.html)

- **Time cost**: 4135.73s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [5e-5,1e-4,5e-4,1e-3,3e-3]
  corruption_ratio choice [0.5,0.3,0.1]
  ```

- **Best parameters**:

  ```yaml
  corruption_ratio: 0.1
  learning_rate: 0.003
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  corruption_ratio:0.5, learning_rate:5e-05
  Valid result:
  recall@10 : 0.0077    mrr@10 : 0.0069    ndcg@10 : 0.0052    hit@10 : 0.0174    precision@10 : 0.0019
  Test result:
  recall@10 : 0.0078    mrr@10 : 0.007    ndcg@10 : 0.0052    hit@10 : 0.0177    precision@10 : 0.0019
  
  corruption_ratio:0.5, learning_rate:0.003
  Valid result:
  recall@10 : 0.0977    mrr@10 : 0.1095    ndcg@10 : 0.0813    hit@10 : 0.1789    precision@10 : 0.0234
  Test result:
  recall@10 : 0.0986    mrr@10 : 0.1111    ndcg@10 : 0.0825    hit@10 : 0.1798    precision@10 : 0.0235
  
  corruption_ratio:0.5, learning_rate:0.0005
  Valid result:
  recall@10 : 0.0077    mrr@10 : 0.0069    ndcg@10 : 0.0052    hit@10 : 0.0175    precision@10 : 0.0019
  Test result:
  recall@10 : 0.0079    mrr@10 : 0.007    ndcg@10 : 0.0052    hit@10 : 0.0178    precision@10 : 0.0019
  
  corruption_ratio:0.3, learning_rate:0.0001
  Valid result:
  recall@10 : 0.0077    mrr@10 : 0.007    ndcg@10 : 0.0052    hit@10 : 0.0175    precision@10 : 0.0019
  Test result:
  recall@10 : 0.0078    mrr@10 : 0.007    ndcg@10 : 0.0052    hit@10 : 0.0177    precision@10 : 0.0019
  
  corruption_ratio:0.3, learning_rate:5e-05
  Valid result:
  recall@10 : 0.0077    mrr@10 : 0.0069    ndcg@10 : 0.0052    hit@10 : 0.0174    precision@10 : 0.0019
  Test result:
  recall@10 : 0.0078    mrr@10 : 0.007    ndcg@10 : 0.0052    hit@10 : 0.0177    precision@10 : 0.0019
  
  corruption_ratio:0.5, learning_rate:0.0001
  Valid result:
  recall@10 : 0.0077    mrr@10 : 0.0069    ndcg@10 : 0.0052    hit@10 : 0.0174    precision@10 : 0.0019
  Test result:
  recall@10 : 0.0078    mrr@10 : 0.007    ndcg@10 : 0.0052    hit@10 : 0.0177    precision@10 : 0.0019
  
  corruption_ratio:0.1, learning_rate:0.0005
  Valid result:
  recall@10 : 0.0076    mrr@10 : 0.0069    ndcg@10 : 0.0052    hit@10 : 0.0174    precision@10 : 0.0019
  Test result:
  recall@10 : 0.0078    mrr@10 : 0.007    ndcg@10 : 0.0052    hit@10 : 0.0177    precision@10 : 0.0019
  
  corruption_ratio:0.5, learning_rate:0.001
  Valid result:
  recall@10 : 0.095    mrr@10 : 0.0838    ndcg@10 : 0.0669    hit@10 : 0.1729    precision@10 : 0.0219
  Test result:
  recall@10 : 0.0957    mrr@10 : 0.0849    ndcg@10 : 0.0677    hit@10 : 0.1743    precision@10 : 0.0222
  
  corruption_ratio:0.3, learning_rate:0.003
  Valid result:
  recall@10 : 0.0954    mrr@10 : 0.1132    ndcg@10 : 0.0824    hit@10 : 0.1761    precision@10 : 0.0232
  Test result:
  recall@10 : 0.0973    mrr@10 : 0.1137    ndcg@10 : 0.0833    hit@10 : 0.1775    precision@10 : 0.0234
  
  corruption_ratio:0.1, learning_rate:0.0001
  Valid result:
  recall@10 : 0.0077    mrr@10 : 0.0069    ndcg@10 : 0.0052    hit@10 : 0.0174    precision@10 : 0.0019
  Test result:
  recall@10 : 0.0078    mrr@10 : 0.007    ndcg@10 : 0.0052    hit@10 : 0.0177    precision@10 : 0.0019
  
  corruption_ratio:0.3, learning_rate:0.001
  Valid result:
  recall@10 : 0.0944    mrr@10 : 0.0856    ndcg@10 : 0.0681    hit@10 : 0.1718    precision@10 : 0.0217
  Test result:
  recall@10 : 0.0938    mrr@10 : 0.0861    ndcg@10 : 0.068    hit@10 : 0.1716    precision@10 : 0.0219
  
  corruption_ratio:0.3, learning_rate:0.0005
  Valid result:
  recall@10 : 0.0077    mrr@10 : 0.0069    ndcg@10 : 0.0052    hit@10 : 0.0175    precision@10 : 0.0019
  Test result:
  recall@10 : 0.0078    mrr@10 : 0.007    ndcg@10 : 0.0052    hit@10 : 0.0177    precision@10 : 0.0019
  
  corruption_ratio:0.1, learning_rate:0.001
  Valid result:
  recall@10 : 0.0899    mrr@10 : 0.0867    ndcg@10 : 0.0673    hit@10 : 0.166    precision@10 : 0.0211
  Test result:
  recall@10 : 0.0919    mrr@10 : 0.0867    ndcg@10 : 0.0679    hit@10 : 0.1686    precision@10 : 0.0215
  
  corruption_ratio:0.1, learning_rate:0.003
  Valid result:
  recall@10 : 0.0937    mrr@10 : 0.121    ndcg@10 : 0.0858    hit@10 : 0.1751    precision@10 : 0.0234
  Test result:
  recall@10 : 0.0946    mrr@10 : 0.1204    ndcg@10 : 0.0859    hit@10 : 0.1754    precision@10 : 0.0234
  
  corruption_ratio:0.1, learning_rate:5e-05
  Valid result:
  recall@10 : 0.0073    mrr@10 : 0.0068    ndcg@10 : 0.005    hit@10 : 0.0167    precision@10 : 0.0018
  Test result:
  recall@10 : 0.0073    mrr@10 : 0.0068    ndcg@10 : 0.005    hit@10 : 0.0168    precision@10 : 0.0018
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 15/15 [17:13:55<00:00, 4135.73s/trial, best loss: -0.0858]
  best params:  {'corruption_ratio': 0.1, 'learning_rate': 0.003}
  best result: 
  {'model': 'CDAE', 'best_valid_score': 0.0858, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0937), ('mrr@10', 0.121), ('ndcg@10', 0.0858), ('hit@10', 0.1751), ('precision@10', 0.0234)]), 'test_result': OrderedDict([('recall@10', 0.0946), ('mrr@10', 0.1204), ('ndcg@10', 0.0859), ('hit@10', 0.1754), ('precision@10', 0.0234)])}
  ```
