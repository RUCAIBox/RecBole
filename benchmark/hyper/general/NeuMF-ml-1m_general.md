# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [NeuMF](https://recbole.io/docs/user_guide/model/general/neumf.html)

- **Time cost**: 403.55s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [1e-4,5e-4,1e-3,5e-3] 
  mlp_hidden_size choice ['[128,64]'] 
  dropout_prob choice [0.1,0.3]                                 
  mf_train choice [True,False]                                             
  mlp_train choice [True,False]
  ```

- **Best parameters**:

  ```yaml
  dropout_prob: 0.1  
  learning_rate: 0.001  
  mf_train: False  
  mlp_train: True
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  dropout_prob:0.1, learning_rate:0.001, mf_train:False, mlp_train:True
  Valid result:
  recall@10 : 0.1476    mrr@10 : 0.3522    ndcg@10 : 0.1914    hit@10 : 0.6784    precision@10 : 0.144
  Test result:
  recall@10 : 0.1611    mrr@10 : 0.4081    ndcg@10 : 0.2287    hit@10 : 0.6945    precision@10 : 0.1725

  dropout_prob:0.3, learning_rate:0.0005, mf_train:True, mlp_train:False
  Valid result:
  recall@10 : 0.148     mrr@10 : 0.3402    ndcg@10 : 0.1863    hit@10 : 0.6778    precision@10 : 0.1403
  Test result:
  recall@10 : 0.1633    mrr@10 : 0.3958    ndcg@10 : 0.2234    hit@10 : 0.696     precision@10 : 0.1688

  dropout_prob:0.3, learning_rate:0.001, mf_train:True, mlp_train:True
  Valid result:
  recall@10 : 0.1423    mrr@10 : 0.3311    ndcg@10 : 0.1807    hit@10 : 0.664     precision@10 : 0.1373
  Test result:
  recall@10 : 0.1566    mrr@10 : 0.3889    ndcg@10 : 0.2188    hit@10 : 0.6869    precision@10 : 0.1662
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 16/16 [1:47:36<00:00, 403.55s/trial, best loss: -0.1914]
  best params:  {'dropout_prob': 0.1, 'learning_rate': 0.001, 'mf_train': False, 'mlp_train': True}
  best result:
  {'model': 'NeuMF', 'best_valid_score': 0.1914, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1476), ('mrr@10', 0.3522), ('ndcg@10', 0.1914), ('hit@10', 0.6784), ('precision@10', 0.144)]), 'test_result': OrderedDict([('recall@10', 0.1611), ('mrr@10', 0.4081), ('ndcg@10', 0.2287), ('hit@10', 0.6945), ('precision@10', 0.1725)])}
  ```
