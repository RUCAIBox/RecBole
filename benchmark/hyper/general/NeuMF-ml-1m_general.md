# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [NeuMF](https://recbole.io/docs/user_guide/model/general/neumf.html)

- **Time cost**: 403.55s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [5e-7,1e-6,5e-6,1e-5,1e-4,1e-3]
  mlp_hidden_size choice ['[64,32,16]']
  dropout_prob choice [0.1,0.0,0.3]
  ```
  
- **Best parameters**:

  ```yaml
  dropout_prob: 0.1  
  learning_rate: 1e-6
  mlp_hidden_size: '[64,32,16]'
  ```
  
- **Hyper-parameter logging** (hyper.result):

  ```yaml
  dropout_prob:0.1, learning_rate:5e-06, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1564    mrr@10 : 0.3563    ndcg@10 : 0.1989    hit@10 : 0.6897    precision@10 : 0.1492
  Test result:
  recall@10 : 0.1723    mrr@10 : 0.4192    ndcg@10 : 0.2392    hit@10 : 0.7149    precision@10 : 0.1795
  
  dropout_prob:0.1, learning_rate:1e-06, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1564    mrr@10 : 0.3571    ndcg@10 : 0.1991    hit@10 : 0.689    precision@10 : 0.1492
  Test result:
  recall@10 : 0.1717    mrr@10 : 0.4195    ndcg@10 : 0.2393    hit@10 : 0.7149    precision@10 : 0.1797
  
  dropout_prob:0.3, learning_rate:5e-07, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1563    mrr@10 : 0.3573    ndcg@10 : 0.1991    hit@10 : 0.6895    precision@10 : 0.1492
  Test result:
  recall@10 : 0.1718    mrr@10 : 0.4197    ndcg@10 : 0.2393    hit@10 : 0.7151    precision@10 : 0.1797
  
  dropout_prob:0.1, learning_rate:1e-05, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1563    mrr@10 : 0.3562    ndcg@10 : 0.1988    hit@10 : 0.6895    precision@10 : 0.1492
  Test result:
  recall@10 : 0.1721    mrr@10 : 0.4192    ndcg@10 : 0.2391    hit@10 : 0.7142    precision@10 : 0.1793
  
  dropout_prob:0.3, learning_rate:1e-06, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1564    mrr@10 : 0.3572    ndcg@10 : 0.1991    hit@10 : 0.6895    precision@10 : 0.1492
  Test result:
  recall@10 : 0.1718    mrr@10 : 0.4197    ndcg@10 : 0.2393    hit@10 : 0.7151    precision@10 : 0.1797
  
  dropout_prob:0.3, learning_rate:0.0001, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1549    mrr@10 : 0.35    ndcg@10 : 0.1943    hit@10 : 0.6874    precision@10 : 0.1456
  Test result:
  recall@10 : 0.1693    mrr@10 : 0.4093    ndcg@10 : 0.2313    hit@10 : 0.7118    precision@10 : 0.1734
  
  dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1556    mrr@10 : 0.3513    ndcg@10 : 0.195    hit@10 : 0.6884    precision@10 : 0.1462
  Test result:
  recall@10 : 0.169    mrr@10 : 0.4101    ndcg@10 : 0.2317    hit@10 : 0.7103    precision@10 : 0.1736
  
  dropout_prob:0.0, learning_rate:5e-06, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1563    mrr@10 : 0.3562    ndcg@10 : 0.1988    hit@10 : 0.6892    precision@10 : 0.1492
  Test result:
  recall@10 : 0.1722    mrr@10 : 0.4193    ndcg@10 : 0.2392    hit@10 : 0.7147    precision@10 : 0.1794
  
  dropout_prob:0.3, learning_rate:1e-05, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1563    mrr@10 : 0.3558    ndcg@10 : 0.1987    hit@10 : 0.6887    precision@10 : 0.1491
  Test result:
  recall@10 : 0.1717    mrr@10 : 0.4192    ndcg@10 : 0.239    hit@10 : 0.7141    precision@10 : 0.1794
  
  dropout_prob:0.3, learning_rate:0.001, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1466    mrr@10 : 0.3371    ndcg@10 : 0.1845    hit@10 : 0.673    precision@10 : 0.1382
  Test result:
  recall@10 : 0.1607    mrr@10 : 0.3901    ndcg@10 : 0.2174    hit@10 : 0.6958    precision@10 : 0.1628
  
  dropout_prob:0.0, learning_rate:1e-05, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1564    mrr@10 : 0.3563    ndcg@10 : 0.1989    hit@10 : 0.6897    precision@10 : 0.1492
  Test result:
  recall@10 : 0.1717    mrr@10 : 0.4191    ndcg@10 : 0.239    hit@10 : 0.7141    precision@10 : 0.1793
  
  dropout_prob:0.0, learning_rate:5e-07, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1564    mrr@10 : 0.3573    ndcg@10 : 0.1991    hit@10 : 0.6897    precision@10 : 0.1493
  Test result:
  recall@10 : 0.1717    mrr@10 : 0.4196    ndcg@10 : 0.2392    hit@10 : 0.7147    precision@10 : 0.1796
  
  dropout_prob:0.0, learning_rate:0.0001, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.156    mrr@10 : 0.352    ndcg@10 : 0.1954    hit@10 : 0.6885    precision@10 : 0.1463
  Test result:
  recall@10 : 0.1687    mrr@10 : 0.411    ndcg@10 : 0.232    hit@10 : 0.7093    precision@10 : 0.1739
  
  dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1476    mrr@10 : 0.3297    ndcg@10 : 0.1814    hit@10 : 0.6743    precision@10 : 0.1357
  Test result:
  recall@10 : 0.1592    mrr@10 : 0.3844    ndcg@10 : 0.2132    hit@10 : 0.6934    precision@10 : 0.159
  
  dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1477    mrr@10 : 0.3292    ndcg@10 : 0.1816    hit@10 : 0.674    precision@10 : 0.1365
  Test result:
  recall@10 : 0.1595    mrr@10 : 0.3797    ndcg@10 : 0.2114    hit@10 : 0.691    precision@10 : 0.1578
  
  dropout_prob:0.1, learning_rate:5e-07, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1564    mrr@10 : 0.3572    ndcg@10 : 0.1991    hit@10 : 0.6895    precision@10 : 0.1493
  Test result:
  recall@10 : 0.1718    mrr@10 : 0.4195    ndcg@10 : 0.2392    hit@10 : 0.7149    precision@10 : 0.1796
  
  dropout_prob:0.0, learning_rate:1e-06, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1564    mrr@10 : 0.3571    ndcg@10 : 0.1991    hit@10 : 0.689    precision@10 : 0.1492
  Test result:
  recall@10 : 0.1717    mrr@10 : 0.4196    ndcg@10 : 0.2393    hit@10 : 0.7149    precision@10 : 0.1797
  
  dropout_prob:0.3, learning_rate:5e-06, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1563    mrr@10 : 0.3568    ndcg@10 : 0.199    hit@10 : 0.6897    precision@10 : 0.1492
  Test result:
  recall@10 : 0.1719    mrr@10 : 0.4195    ndcg@10 : 0.2393    hit@10 : 0.7152    precision@10 : 0.1797
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 18/18 [21:15<00:00, 70.88s/trial, best loss: -0.1991]
  best params:  {'dropout_prob': 0.1, 'learning_rate': 1e-06, 'mlp_hidden_size': '[64,32,16]'}
  best result: 
  {'model': 'NeuMF', 'best_valid_score': 0.1991, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1564), ('mrr@10', 0.3571), ('ndcg@10', 0.1991), ('hit@10', 0.689), ('precision@10', 0.1492)]), 'test_result': OrderedDict([('recall@10', 0.1717), ('mrr@10', 0.4195), ('ndcg@10', 0.2393), ('hit@10', 0.7149), ('precision@10', 0.1797)])}
  ```
