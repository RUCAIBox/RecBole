# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [ADMMSLIM](https://recbole.io/docs/user_guide/model/general/admmslim.html)

- **Time cost**: 308.15s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  alpha choice [0.25,0.5,0.75,1]
  lambda1 choice [0.1,0.5,5,10]
  lambda2 choice [5,50,1000,5000]
  ```

- **Best parameters**:

  ```yaml
  alpha: 1
  lambda1: 5
  lambda2: 5000
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  alpha:0.5, lambda1:0.1, lambda2:5000
  Valid result:
  recall@10 : 0.1869    mrr@10 : 0.4031    ndcg@10 : 0.2304    hit@10 : 0.7504    precision@10 : 0.1666
  Test result:
  recall@10 : 0.2094    mrr@10 : 0.4879    ndcg@10 : 0.2853    hit@10 : 0.7752    precision@10 : 0.2051

  alpha:0.5, lambda1:5, lambda2:5000
  Valid result:
  recall@10 : 0.1861    mrr@10 : 0.4031    ndcg@10 : 0.2298    hit@10 : 0.7479    precision@10 : 0.1661
  Test result:
  recall@10 : 0.2086    mrr@10 : 0.4855    ndcg@10 : 0.2835    hit@10 : 0.7757    precision@10 : 0.204
  
  alpha:1, lambda1:0.5, lambda2:5000
  Valid result:
  recall@10 : 0.1907    mrr@10 : 0.4053    ndcg@10 : 0.2344    hit@10 : 0.7544    precision@10 : 0.1704
  Test result:
  recall@10 : 0.2139    mrr@10 : 0.4909    ndcg@10 : 0.2903    hit@10 : 0.7802    precision@10 : 0.2099
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 64/64 [5:28:41<00:00, 308.15s/trial, best loss: -0.2356]
  best params:  {'alpha': 1, 'lambda1': 5, 'lambda2': 5000}
  best result: 
  {'model': 'ADMMSLIM', 'best_valid_score': 0.2356, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1935), ('mrr@10', 0.4052), ('ndcg@10', 0.2356), ('hit@10', 0.7567), ('precision@10', 0.1718)]), 'test_result': OrderedDict([('recall@10', 0.2134), ('mrr@10', 0.49), ('ndcg@10', 0.2904), ('hit@10', 0.7804), ('precision@10', 0.2106)])}
  ```
