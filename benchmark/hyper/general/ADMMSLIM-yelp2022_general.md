# General Recommendation

- **Dataset**: [Yelp2022](../../md/yelp_general.md)

- **Model**: [ADMMSLIM](https://recbole.io/docs/user_guide/model/general/admmslim.html)

- **Time cost**: 148140.63s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  alpha choice [0.5,1]
  lambda1 choice [0.5,5]
  lambda2 choice [1000, 5000]
  ```

- **Best parameters**:

  ```yaml
  alpha: 0.5
  lambda1: 0.5
  lambda2: 1000
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  alpha: 0.5, lambda1: 5, lambda2: 1000
  Valid result:
  recall@10 : 0.0758    mrr@10 : 0.0652    ndcg@10 : 0.0513    hit@10 : 0.1498    precision@10 : 0.0189
  Test result:
  recall@10 : 0.0774    mrr@10 : 0.0674    ndcg@10 : 0.0527    hit@10 : 0.1518    precision@10 : 0.0192
  
  alpha:1, lambda1:0.5, lambda2:1000
  Valid result:
  recall@10 : 0.0955    mrr@10 : 0.0929    ndcg@10 : 0.0696    hit@10 : 0.1872    precision@10 : 0.026
  Test result:
  recall@10 : 0.0976    mrr@10 : 0.0956    ndcg@10 : 0.0717    hit@10 : 0.19    precision@10 : 0.0266
  
  alpha: 0.5, lambda1:0.5, lambda2:1000
  Valid result:
  recall@10 : 0.0969    mrr@10 : ,0.0938    ndcg@10 : 0.0705    hit@10 : 0.1893    precision@10 : 0.0262
  Test result:
  recall@10 : 0.1006    mrr@10 : 0.0968    ndcg@10 : 0.0731    hit@10 : 0.1936    precision@10 : 0.0269
  ```
  
- **Logging Result**:

  ```yaml
  100%|█████████| 8/8 [325:57:24<00:00:00, 148140.63s/trial, best loss: -0.0705]
  best params:  {'alpha': 0.5, 'lambda1': 0.5, 'lambda2': 1000}
  best result: 
  {'model': 'ADMMSLIM', 'best_valid_score': 0.0705, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0969), ('mrr@10',0.0938), ('ndcg@10', 0.0705), ('hit@10', 0.1893), ('precision@10', 0.0262)]), 'test_result': OrderedDict([('recall@10', 0.1006), ('mrr@10', 0.0968), ('ndcg@10', 0.0731), ('hit@10', 0.1936), ('precision@10', 0.0269)])}
  ```
