# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [GCMC](https://recbole.io/docs/user_guide/model/general/gcmc.html)

- **Time cost**: 1403.44s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  accum choice ['stack','sum'] 
  learning_rate choice [0.001,0.005,0.01] 
  dropout_prob choice [0.3,0.5,0.7] 
  gcn_output_dim choice [500] 
  num_basis_functions choice [2]
  ```

- **Best parameters**:

  ```yaml
  accum: 'stack'  
  dropout_prob: 0.5  
  gcn_output_dim: 500  
  num_basis_functions: 2  
  learning_rate: 1e-3
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  accum:'stack', dropout_prob:0.5, gcn_output_dim:500, num_basis_functions:2, learning_rate:0.001
  Valid result:
  recall@10 : 0.1609    mrr@10 : 0.3604    ndcg@10 : 0.2006    hit@10 : 0.7006    precision@10 : 0.1495
  Test result:
  recall@10 : 0.1772    mrr@10 : 0.4246    ndcg@10 : 0.2442    hit@10 : 0.7214    precision@10 : 0.1823

  accum:'stack', dropout_prob:0.7, gcn_output_dim:500, num_basis_functions:2, learning_rate:0.005
  Valid result:
  recall@10 : 0.116     mrr@10 : 0.3045    ndcg@10 : 0.1568    hit@10 : 0.6007    precision@10 : 0.118
  Test result:
  recall@10 : 0.1256    mrr@10 : 0.3571    ndcg@10 : 0.1861    hit@10 : 0.6186    precision@10 : 0.1397

  accum:'sum', dropout_prob:0.5, gcn_output_dim:500, num_basis_functions:2, learning_rate:0.005
  Valid result:
  recall@10 : 0.1289    mrr@10 : 0.326     ndcg@10 : 0.172     hit@10 : 0.6309    precision@10 : 0.1302
  Test result:
  recall@10 : 0.1396    mrr@10 : 0.3779    ndcg@10 : 0.2041    hit@10 : 0.6534    precision@10 : 0.1542
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 18/18 [7:01:02<00:00, 1403.44s/trial, best loss: -0.2006]
  best params:  {'accum': 'stack', 'dropout_prob': 0.5, 'gcn_output_dim': 500, 'learning_rate': 0.001, 'num_basis_functions': 2}
  best result: 
  {'model': 'GCMC', 'best_valid_score': 0.2006, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1609), ('mrr@10', 0.3604), ('ndcg@10', 0.2006), ('hit@10', 0.7006), ('precision@10', 0.1495)]), 'test_result': OrderedDict([('recall@10', 0.1772), ('mrr@10', 0.4246), ('ndcg@10', 0.2442), ('hit@10', 0.7214), ('precision@10', 0.1823)])}
  ```
