# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [NNCF](https://recbole.io/docs/user_guide/model/general/nncf.html)

- **Time cost**: 4895.53s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  neigh_num choice [20,50,100]
  neigh_embedding_size choice [64,32]
  num_conv_kernel choice [128,64]
  learning_rate choice [5e-5,1e-4,5e-4]
  neigh_info_method choice ['random','knn']
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0001  
  neigh_embedding_size: 64  
  num_conv_kernel: 128  
  neigh_num: 20  
  neigh_info_method: 'random'
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0001, neigh_embedding_size:64, neigh_info_method:'random', neigh_num:100, num_conv_kernel:128
  Valid result:
  recall@10 : 0.1503    mrr@10 : 0.3588    ndcg@10 : 0.1961    hit@10 : 0.6867    precision@10 : 0.1468
  Test result:
  recall@10 : 0.1664    mrr@10 : 0.419     ndcg@10 : 0.2377    hit@10 : 0.7069    precision@10 : 0.1788

  learning_rate:5e-05, neigh_embedding_size:32, neigh_info_method:'knn', neigh_num:100, num_conv_kernel:128
  Valid result:
  recall@10 : 0.1535    mrr@10 : 0.3624    ndcg@10 : 0.1984    hit@10 : 0.6907    precision@10 : 0.1484
  Test result:
  recall@10 : 0.1678    mrr@10 : 0.4272    ndcg@10 : 0.2406    hit@10 : 0.1096    precision@10 : 0.18

  learning_rate:5e-05, neigh_embedding_size:32, neigh_info_method:'random', neigh_num:50, num_conv_kernel:64
  Valid result:
  recall@10 : 0.1571    mrr@10 : 0.3683    ndcg@10 : 0.2018    hit@10 : 0.7016    precision@10 : 0.1504
  Test result:
  recall@10 : 0.1745    mrr@10 : 0.4296    ndcg@10 : 0.2454    hit@10 : 0.7182    precision@10 : 0.1835
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 72/72 [97:54:38<00:00, 4895.53s/trial, best loss: -0.2034]
  best params:  {'learning_rate': 0.0001, 'neigh_embedding_size': 64, 'neigh_info_method': 'random', 'neigh_num': 20, 'num_conv_kernel': 128}
  best result: 
  {'model': 'NNCF', 'best_valid_score': 0.2034, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1605), ('mrr@10', 0.3678), ('ndcg@10', 0.2034), ('hit@10', 0.7033), ('precision@10', 0.1518)]), 'test_result': OrderedDict([('recall@10', 0.1753), ('mrr@10', 0.4301), ('ndcg@10', 0.2468), ('hit@10', 0.7212), ('precision@10', 0.1854)])}
  ```
