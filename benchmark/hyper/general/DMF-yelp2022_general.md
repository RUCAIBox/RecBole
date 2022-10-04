# General Recommendation

- **Dataset**: [Yelp2022](../../md/yelp_general.md)

- **Model**: [DMF](https://recbole.io/docs/user_guide/model/general/dmf.html)

- **Time cost**: 6070.33s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [5e-5,1e-4,5e-4,1e-3,5e-3]
  user_layers_dim choice ['[64, 64]','[64, 32]']
  item_layers_dim choice ['[64, 64]','[64, 32]']
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  user_layers_dim: '[64,32]'
  item_layers_dim: '[64,64]'
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  item_layers_dim:[64,32], learning_rate:5e-05, user_layers_dim:[64,64]
  Valid result:
  recall@10 : 0.0368    mrr@10 : 0.027    ndcg@10 : 0.0227    hit@10 : 0.0739    precision@10 : 0.0081
  Test result:
  recall@10 : 0.0381    mrr@10 : 0.0282    ndcg@10 : 0.0234    hit@10 : 0.0757    precision@10 : 0.0083
  
  item_layers_dim:[64,32], learning_rate:5e-05, user_layers_dim:[64,32]
  Valid result:
  recall@10 : 0.0368    mrr@10 : 0.027    ndcg@10 : 0.0227    hit@10 : 0.0739    precision@10 : 0.0081
  Test result:
  recall@10 : 0.0381    mrr@10 : 0.0282    ndcg@10 : 0.0234    hit@10 : 0.0757    precision@10 : 0.0083
  
  item_layers_dim:[64,32], learning_rate:0.0001, user_layers_dim:[64,64]
  Valid result:
  recall@10 : 0.0385    mrr@10 : 0.0274    ndcg@10 : 0.0235    hit@10 : 0.0768    precision@10 : 0.0084
  Test result:
  recall@10 : 0.0402    mrr@10 : 0.0279    ndcg@10 : 0.0239    hit@10 : 0.0795    precision@10 : 0.0086
  
  item_layers_dim:[64,64], learning_rate:0.0001, user_layers_dim:[64,64]
  Valid result:
  recall@10 : 0.0385    mrr@10 : 0.0274    ndcg@10 : 0.0235    hit@10 : 0.0768    precision@10 : 0.0084
  Test result:
  recall@10 : 0.0402    mrr@10 : 0.0279    ndcg@10 : 0.0239    hit@10 : 0.0795    precision@10 : 0.0086
  
  item_layers_dim:[64,64], learning_rate:0.001, user_layers_dim:[64,32]
  Valid result:
  recall@10 : 0.0446    mrr@10 : 0.032    ndcg@10 : 0.0272    hit@10 : 0.0887    precision@10 : 0.0098
  Test result:
  recall@10 : 0.0448    mrr@10 : 0.032    ndcg@10 : 0.027    hit@10 : 0.0885    precision@10 : 0.0098
  
  item_layers_dim:[64,64], learning_rate:5e-05, user_layers_dim:[64,64]
  Valid result:
  recall@10 : 0.0368    mrr@10 : 0.027    ndcg@10 : 0.0227    hit@10 : 0.0739    precision@10 : 0.0081
  Test result:
  recall@10 : 0.0381    mrr@10 : 0.0282    ndcg@10 : 0.0234    hit@10 : 0.0757    precision@10 : 0.0083
  
  item_layers_dim:[64,32], learning_rate:0.001, user_layers_dim:[64,32]
  Valid result:
  recall@10 : 0.0446    mrr@10 : 0.032    ndcg@10 : 0.0272    hit@10 : 0.0887    precision@10 : 0.0098
  Test result:
  recall@10 : 0.0448    mrr@10 : 0.032    ndcg@10 : 0.027    hit@10 : 0.0885    precision@10 : 0.0098
  
  item_layers_dim:[64,32], learning_rate:0.0001, user_layers_dim:[64,32]
  Valid result:
  recall@10 : 0.0385    mrr@10 : 0.0274    ndcg@10 : 0.0235    hit@10 : 0.0768    precision@10 : 0.0084
  Test result:
  recall@10 : 0.0402    mrr@10 : 0.0279    ndcg@10 : 0.0239    hit@10 : 0.0795    precision@10 : 0.0086
  
  item_layers_dim:[64,64], learning_rate:0.0001, user_layers_dim:[64,32]
  Valid result:
  recall@10 : 0.0385    mrr@10 : 0.0274    ndcg@10 : 0.0235    hit@10 : 0.0768    precision@10 : 0.0084
  Test result:
  recall@10 : 0.0402    mrr@10 : 0.0279    ndcg@10 : 0.0239    hit@10 : 0.0795    precision@10 : 0.0086
  
  item_layers_dim:[64,32], learning_rate:0.001, user_layers_dim:[64,64]
  Valid result:
  recall@10 : 0.0446    mrr@10 : 0.032    ndcg@10 : 0.0272    hit@10 : 0.0887    precision@10 : 0.0098
  Test result:
  recall@10 : 0.0448    mrr@10 : 0.032    ndcg@10 : 0.027    hit@10 : 0.0885    precision@10 : 0.0098
  
  item_layers_dim:[64,64], learning_rate:5e-05, user_layers_dim:[64,32]
  Valid result:
  recall@10 : 0.0368    mrr@10 : 0.027    ndcg@10 : 0.0227    hit@10 : 0.0739    precision@10 : 0.0081
  Test result:
  recall@10 : 0.0381    mrr@10 : 0.0282    ndcg@10 : 0.0234    hit@10 : 0.0757    precision@10 : 0.0083
  
  item_layers_dim:[64,32], learning_rate:0.005, user_layers_dim:[64,64]
  Valid result:
  recall@10 : 0.0349    mrr@10 : 0.0268    ndcg@10 : 0.022    hit@10 : 0.0712    precision@10 : 0.0078
  Test result:
  recall@10 : 0.036    mrr@10 : 0.0272    ndcg@10 : 0.0224    hit@10 : 0.0722    precision@10 : 0.0079
  
  item_layers_dim:[64,32], learning_rate:0.0005, user_layers_dim:[64,32]
  Valid result:
  recall@10 : 0.0417    mrr@10 : 0.0302    ndcg@10 : 0.0255    hit@10 : 0.0838    precision@10 : 0.0093
  Test result:
  recall@10 : 0.0437    mrr@10 : 0.0303    ndcg@10 : 0.026    hit@10 : 0.0861    precision@10 : 0.0096
  
  item_layers_dim:[64,64], learning_rate:0.005, user_layers_dim:[64,32]
  Valid result:
  recall@10 : 0.0349    mrr@10 : 0.0268    ndcg@10 : 0.022    hit@10 : 0.0712    precision@10 : 0.0078
  Test result:
  recall@10 : 0.036    mrr@10 : 0.0272    ndcg@10 : 0.0224    hit@10 : 0.0722    precision@10 : 0.0079
  
  item_layers_dim:[64,32], learning_rate:0.0005, user_layers_dim:[64,64]
  Valid result:
  recall@10 : 0.0417    mrr@10 : 0.0302    ndcg@10 : 0.0255    hit@10 : 0.0838    precision@10 : 0.0093
  Test result:
  recall@10 : 0.0437    mrr@10 : 0.0303    ndcg@10 : 0.026    hit@10 : 0.0861    precision@10 : 0.0096
  
  item_layers_dim:[64,64], learning_rate:0.005, user_layers_dim:[64,64]
  Valid result:
  recall@10 : 0.0349    mrr@10 : 0.0268    ndcg@10 : 0.022    hit@10 : 0.0712    precision@10 : 0.0078
  Test result:
  recall@10 : 0.036    mrr@10 : 0.0272    ndcg@10 : 0.0224    hit@10 : 0.0722    precision@10 : 0.0079
  
  item_layers_dim:[64,64], learning_rate:0.001, user_layers_dim:[64,64]
  Valid result:
  recall@10 : 0.0446    mrr@10 : 0.032    ndcg@10 : 0.0272    hit@10 : 0.0887    precision@10 : 0.0098
  Test result:
  recall@10 : 0.0448    mrr@10 : 0.032    ndcg@10 : 0.027    hit@10 : 0.0885    precision@10 : 0.0098
  
  item_layers_dim:[64,32], learning_rate:0.005, user_layers_dim:[64,32]
  Valid result:
  recall@10 : 0.0349    mrr@10 : 0.0268    ndcg@10 : 0.022    hit@10 : 0.0712    precision@10 : 0.0078
  Test result:
  recall@10 : 0.036    mrr@10 : 0.0272    ndcg@10 : 0.0224    hit@10 : 0.0722    precision@10 : 0.0079
  
  item_layers_dim:[64,64], learning_rate:0.0005, user_layers_dim:[64,64]
  Valid result:
  recall@10 : 0.0417    mrr@10 : 0.0302    ndcg@10 : 0.0255    hit@10 : 0.0838    precision@10 : 0.0093
  Test result:
  recall@10 : 0.0437    mrr@10 : 0.0303    ndcg@10 : 0.026    hit@10 : 0.0861    precision@10 : 0.0096
  
  item_layers_dim:[64,64], learning_rate:0.0005, user_layers_dim:[64,32]
  Valid result:
  recall@10 : 0.0417    mrr@10 : 0.0302    ndcg@10 : 0.0255    hit@10 : 0.0838    precision@10 : 0.0093
  Test result:
  recall@10 : 0.0437    mrr@10 : 0.0303    ndcg@10 : 0.026    hit@10 : 0.0861    precision@10 : 0.0096
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 20/20 [33:43:26<00:00, 6070.33s/trial, best loss: -0.0272]
  best params:  {'item_layers_dim': '[64,64]', 'learning_rate': 0.001, 'user_layers_dim': '[64,32]'}
  best result: 
  {'model': 'DMF', 'best_valid_score': 0.0272, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0446), ('mrr@10', 0.032), ('ndcg@10', 0.0272), ('hit@10', 0.0887), ('precision@10', 0.0098)]), 'test_result': OrderedDict([('recall@10', 0.0448), ('mrr@10', 0.032), ('ndcg@10', 0.027), ('hit@10', 0.0885), ('precision@10', 0.0098)])}
  ```
