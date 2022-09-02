# Sequential Recommendation

- **Dataset**: [Yelp](../../md/yelp_seq.md)

- **Model**: [CORE](https://recbole.io/docs/user_guide/model/sequential/core.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.005,0.001,0.0005,0.0001]
  weight choice [0.5,0.6]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0001
  weight: 0.5
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0001, weight:0.6
  Valid result:
  recall@10 : 0.0706    mrr@10 : 0.0241    ndcg@10 : 0.0348    hit@10 : 0.0706    precision@10 : 0.0071
  Test result:
  recall@10 : 0.0691    mrr@10 : 0.0239    ndcg@10 : 0.0343    hit@10 : 0.0691    precision@10 : 0.0069
  
  learning_rate:0.005, weight:0.5
  Valid result:
  recall@10 : 0.0691    mrr@10 : 0.0234    ndcg@10 : 0.0339    hit@10 : 0.0691    precision@10 : 0.0069
  Test result:
  recall@10 : 0.0672    mrr@10 : 0.023    ndcg@10 : 0.0332    hit@10 : 0.0672    precision@10 : 0.0067
  
  learning_rate:0.001, weight:0.6
  Valid result:
  recall@10 : 0.072    mrr@10 : 0.0245    ndcg@10 : 0.0354    hit@10 : 0.072    precision@10 : 0.0072
  Test result:
  recall@10 : 0.07    mrr@10 : 0.0242    ndcg@10 : 0.0347    hit@10 : 0.07    precision@10 : 0.007
  
  learning_rate:0.005, weight:0.6
  Valid result:
  recall@10 : 0.0677    mrr@10 : 0.023    ndcg@10 : 0.0333    hit@10 : 0.0677    precision@10 : 0.0068
  Test result:
  recall@10 : 0.0655    mrr@10 : 0.0225    ndcg@10 : 0.0324    hit@10 : 0.0655    precision@10 : 0.0065
  
  learning_rate:0.001, weight:0.5
  Valid result:
  recall@10 : 0.0677    mrr@10 : 0.023    ndcg@10 : 0.0333    hit@10 : 0.0677    precision@10 : 0.0068
  Test result:
  recall@10 : 0.0655    mrr@10 : 0.0225    ndcg@10 : 0.0324    hit@10 : 0.0655    precision@10 : 0.0065
  
  learning_rate:0.0005, weight:0.5
  Valid result:
  recall@10 : 0.0691    mrr@10 : 0.0234    ndcg@10 : 0.0339    hit@10 : 0.0691    precision@10 : 0.0069
  Test result:
  recall@10 : 0.0672    mrr@10 : 0.023    ndcg@10 : 0.0332    hit@10 : 0.0672    precision@10 : 0.0067
  
  learning_rate:0.0001, weight:0.5
  Valid result:
  recall@10 : 0.072    mrr@10 : 0.0245    ndcg@10 : 0.0354    hit@10 : 0.072    precision@10 : 0.0072
  Test result:
  recall@10 : 0.07    mrr@10 : 0.0242    ndcg@10 : 0.0347    hit@10 : 0.07    precision@10 : 0.007
  
  learning_rate:0.0005, weight:0.6
  Valid result:
  recall@10 : 0.0706    mrr@10 : 0.0241    ndcg@10 : 0.0348    hit@10 : 0.0706    precision@10 : 0.0071
  Test result:
  recall@10 : 0.0691    mrr@10 : 0.0239    ndcg@10 : 00343    hit@10 : 0.0691    precision@10 : 0.0069
  ```

- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.0001, 'weight': 0.5}
  best result: 
  {'model': 'CORE', 'best_valid_result': OrderedDict([('recall@10', 0.072), ('mrr@10', 0.0245), ('ndcg@10', 0.0354), ('hit@10', 0.072), ('precision@10', 0.0072)]), 'test_result': OrderedDict([('recall@10', 0.0691), ('mrr@10', 0.0239), ('ndcg@10', 0.0343), ('hit@10', 0.0691), ('precision@10', 0.0069)])}
  ```
