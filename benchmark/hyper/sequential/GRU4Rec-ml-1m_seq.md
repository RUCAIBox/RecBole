# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [GRU4Rec](https://recbole.io/docs/user_guide/model/sequential/gru4rec.html)

- **Time cost**: 2188.99s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.005,0.001,0.0005,0.0001]
  num_layers choice [1,2]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.005
  num_layers: 2
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0001, num_layers:2
  Valid result:
  recall@10 : 0.2712    mrr@10 : 0.1088    ndcg@10 : 0.1466    hit@10 : 0.2712    precision@10 : 0.0271
  Test result:
  recall@10 : 0.2546    mrr@10 : 0.1033    ndcg@10 : 0.1385    hit@10 : 0.2546    precision@10 : 0.0255
  
  learning_rate:0.005, num_layers:1
  Valid result:
  recall@10 : 0.2907    mrr@10 : 0.1263    ndcg@10 : 0.1647    hit@10 : 0.2907    precision@10 : 0.0291
  Test result:
  recall@10 : 0.2753    mrr@10 : 0.1185    ndcg@10 : 0.1551    hit@10 : 0.2753    precision@10 : 0.0275
  
  learning_rate:0.001, num_layers:2
  Valid result:
  recall@10 : 0.2954    mrr@10 : 0.1243    ndcg@10 : 0.1643    hit@10 : 0.2954    precision@10 : 0.0295
  Test result:
  recall@10 : 0.2748    mrr@10 : 0.1189    ndcg@10 : 0.1551    hit@10 : 0.2748    precision@10 : 0.0275
  
  learning_rate:0.005, num_layers:2
  Valid result:
  recall@10 : 0.3027    mrr@10 : 0.1263    ndcg@10 : 0.1674    hit@10 : 0.3027    precision@10 : 0.0303
  Test result:
  recall@10 : 0.2776    mrr@10 : 0.1171    ndcg@10 : 0.1545    hit@10 : 0.2776    precision@10 : 0.0278
  
  learning_rate:0.001, num_layers:1
  Valid result:
  recall@10 : 0.2844    mrr@10 : 0.119    ndcg@10 : 0.1576    hit@10 : 0.2844    precision@10 : 0.0284
  Test result:
  recall@10 : 0.2685    mrr@10 : 0.1113    ndcg@10 : 0.148    hit@10 : 0.2685    precision@10 : 0.0269
  
  learning_rate:0.0005, num_layers:1
  Valid result:
  recall@10 : 0.2836    mrr@10 : 0.1188    ndcg@10 : 0.157    hit@10 : 0.2836    precision@10 : 0.0284
  Test result:
  recall@10 : 0.2631    mrr@10 : 0.1089    ndcg@10 : 0.1448    hit@10 : 0.2631    precision@10 : 0.0263
  
  learning_rate:0.0001, num_layers:1
  Valid result:
  recall@10 : 0.2515    mrr@10 : 0.1007    ndcg@10 : 0.1357    hit@10 : 0.2515    precision@10 : 0.0251
  Test result:
  recall@10 : 0.2398    mrr@10 : 0.0966    ndcg@10 : 0.1301    hit@10 : 0.2398    precision@10 : 0.024
  
  learning_rate:0.0005, num_layers:2
  Valid result:
  recall@10 : 0.2891    mrr@10 : 0.1195    ndcg@10 : 0.159    hit@10 : 0.2891    precision@10 : 0.0289
  Test result:
  recall@10 : 0.2677    mrr@10 : 0.1145    ndcg@10 : 0.1501    hit@10 : 0.2677    precision@10 : 0.0268
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 8/8 [4:51:51<00:00, 2188.99s/trial, best loss: -0.1674]
  best params:  {'learning_rate': 0.005, 'num_layers': 2}
  best result: 
  {'model': 'GRU4Rec', 'best_valid_score': 0.1674, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.3027), ('mrr@10', 0.1263), ('ndcg@10', 0.1674), ('hit@10', 0.3027), ('precision@10', 0.0303)]), 'test_result': OrderedDict([('recall@10', 0.2776), ('mrr@10', 0.1171), ('ndcg@10', 0.1545), ('hit@10', 0.2776), ('precision@10', 0.0278)])}
  ```
