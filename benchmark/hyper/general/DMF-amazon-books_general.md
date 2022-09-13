# General Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_general.md)

- **Model**: [DMF](https://recbole.io/docs/user_guide/model/general/dmf.html)

- **Time cost**: 903.73s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate in [5e-5,5e-4,3e-4,1e-4,5e-3,1e-3] 
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0005
  Valid result:
  recall@10 : 0.0506    mrr@10 : 0.0371    ndcg@10 : 0.0314    hit@10 : 0.0997    precision@10 : 0.0116
  Test result:
  recall@10 : 0.0519    mrr@10 : 0.0368    ndcg@10 : 0.0318    hit@10 : 0.1009    precision@10 : 0.012

  learning_rate:0.001
  Valid result:
  recall@10 : 0.0502    mrr@10 : 0.036    ndcg@10 : 0.031    hit@10 : 0.0988    precision@10 : 0.0114
  Test result:
  recall@10 : 0.0504    mrr@10 : 0.0362    ndcg@10 : 0.0309    hit@10 : 0.0986    precision@10 : 0.0116

  learning_rate:0.0001
  Valid result:
  recall@10 : 0.0431    mrr@10 : 0.029    ndcg@10 : 0.0248    hit@10 : 0.0877    precision@10 : 0.01
  Test result:
  recall@10 : 0.0428    mrr@10 : 0.0289    ndcg@10 : 0.0249    hit@10 : 0.0854    precision@10 : 0.01

  learning_rate:0.005
  Valid result:
  recall@10 : 0.0343    mrr@10 : 0.0261    ndcg@10 : 0.0219    hit@10 : 0.0698    precision@10 : 0.0081
  Test result:
  recall@10 : 0.0346    mrr@10 : 0.0259    ndcg@10 : 0.022    hit@10 : 0.0705    precision@10 : 0.0084

  learning_rate:5e-05
  Valid result:
  recall@10 : 0.0118    mrr@10 : 0.0086    ndcg@10 : 0.0083    hit@10 : 0.0192    precision@10 : 0.002
  Test result:
  recall@10 : 0.0105    mrr@10 : 0.0077    ndcg@10 : 0.0072    hit@10 : 0.018    precision@10 : 0.0018

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.0485    mrr@10 : 0.0351    ndcg@10 : 0.0299    hit@10 : 0.0956    precision@10 : 0.011
  Test result:
  recall@10 : 0.0483    mrr@10 : 0.0351    ndcg@10 : 0.0299    hit@10 : 0.0952    precision@10 : 0.0111
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 6/6 [1:30:22<00:00, 903.73s/trial, best loss: -0.0314]
  best params:  {'learning_rate': 0.0005}
  best result: 
  {'model': 'DMF', 'best_valid_score': 0.0314, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0506), ('mrr@10', 0.0371), ('ndcg@10', 0.0314), ('hit@10', 0.0997), ('precision@10', 0.0116)]), 'test_result': OrderedDict([('recall@10', 0.0519), ('mrr@10', 0.0368), ('ndcg@10', 0.0318), ('hit@10', 0.1009), ('precision@10', 0.012)])}
  ```
